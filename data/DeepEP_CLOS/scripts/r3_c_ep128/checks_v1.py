from bench_v1_precision import *

def check_case(args, buffer):
    assert args.precision_only, 'Batch checker must not run performance'
    sys.path.insert(0, str(REPO / 'tests/clos'))
    import torch
    import torch.distributed as dist
    import torch_npu
    from deep_ep.backends.global_clos import Buffer
    from deep_ep import topk_idx_t
    from deep_ep.backends.direct import Config
    import common as golden
    import profiling
    rank, local_rank = (int(os.environ['RANK']), int(os.environ['LOCAL_RANK']))
    if int(os.environ['WORLD_SIZE']) != args.ep_size or int(os.environ['LOCAL_WORLD_SIZE']) != shared.local_world_size(args) or (not 0 <= local_rank < shared.local_world_size(args)) or (rank != args.node_rank * shared.local_world_size(args) + local_rank):
        raise RuntimeError('torchrun local/global rank mapping does not match this node')
    args.num_aiv, args.dtype, expanded = (args.aiv[0], args.dtypes[0], bool(args.expand[0]))
    args.expert_alignment, args.zero_copy = (1, False)
    args.profile_warmup, args.profile_iters = (args.warmup, args.iterations)
    args.profile_sample_markers = True
    args.combine_notify = True
    os.environ[shared.COMBINE_NOTIFY_ENV] = '0'
    device = args.first_device + local_rank
    destination = args.output / f'rank{rank}'
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / 'process.json', dict(pid=os.getpid(), rank=rank, device=device, node_rank=args.node_rank, started=time.time()))
    try:
        torch.set_num_threads(1)
        routes = [golden._make_topk_idx(p, args) for p in range(args.ep_size)]
        counts = route_counts([r.tolist() for r in routes], args.experts_per_rank)
        write_json(destination / 'route_load.json', route_statistics([r.tolist() for r in routes], args.experts_per_rank, args.route_group_size))
        write_json(destination / 'input_routes.json', routes[rank].tolist())
        torch.npu.set_device(device)
        config = Config(args.num_aiv)
        args.num_bytes = config.get_nvl_buffer_size_hint(args.hidden * 2, args.ep_size, num_tokens=args.num_tokens)
        args.resolved_udma_policy = asdict(buffer.udma_auto_policy.for_launch(config.num_aiv, args.hidden * 2, 'dispatch', v2=False))
        args.effective_udma_config = shared.effective_udma_config(buffer, config, row_bytes=args.hidden * 2, v2=False)
        if rank == 0:
            print('EFFECTIVE_UDMA_CONFIG', json.dumps(args.effective_udma_config, sort_keys=True), flush=True)
        identity = shared.loaded_identity()
        identities = [None] * args.ep_size
        dist.all_gather_object(identities, library_identity(identity))
        if any((item != identities[0] for item in identities)):
            raise RuntimeError('ranks loaded different library bytes or Python modules')
        idx = routes[rank].to('npu')
        x = golden._to_npu(golden._make_dispatch_x(rank, args))
        cached_x = golden._to_npu(golden._make_dispatch_x(rank, args, delta=1))
        weights = golden._make_weights(rank, args).to('npu')
        x_parts = x if isinstance(x, tuple) else (x,)
        input_bytes = sum((t.numel() * t.element_size() for t in (*x_parts, idx, weights)))
        ledger = byte_ledger(counts, rank, tokens=args.num_tokens, hidden=args.hidden, topk=args.num_topk, dtype=args.dtype, idx_bytes=idx.element_size(), op='dispatch')
        if input_bytes != args.num_tokens * ledger['official_logical_bytes_per_row']:
            raise ValueError('logical row-byte formula does not match actual dispatch input tensors')
        operations = V1Operations(buffer, config, idx, weights, args.ep_size * args.experts_per_rank, expanded)

        def wait(result):
            profiling._consume_profile_result(result)
            return result

        def check_dispatch(result, delta=0, cached=False, check_storage=True):
            dispatched, expansion = result[:2]
            ref = golden._dispatch_reference(rank, args, delta=delta)
            golden._assert_equal('dispatch payload/scales', dispatched[0], ref['recv_x'])
            if check_storage:
                golden._assert_recv_storage('dispatch payload', dispatched[0], buffer, False)
            if not cached:
                golden._assert_equal('dispatch indices', dispatched[1], ref['recv_topk_idx'])
                golden._assert_equal('dispatch weights', dispatched[2], ref['recv_topk_weights'])
                golden._assert_equal('expert counts', torch.tensor(dispatched[3], dtype=torch.int32), ref['expert_counts'])
                for name, value, wanted in [('rank prefix', dispatched[4][0], ref['rank_prefix']), ('local prefix', dispatched[4][1], ref['local_channel_prefix']), ('recv prefix', dispatched[4][2], ref['recv_channel_prefix']), ('src_idx', dispatched[4][3], ref['recv_src_idx']), ('send_head', dispatched[4][5], ref['send_head'])]:
                    golden._assert_equal(name, value, wanted)
            elif any((value is not None for value in dispatched[1:5])):
                raise AssertionError('payload-only cached dispatch returned metadata')
            raw_x = dispatched[0][0] if isinstance(dispatched[0], tuple) else dispatched[0]
            if raw_x.size(0) != sum((counts['rows'][p][rank] for p in range(args.ep_size))):
                raise AssertionError('independent route count disagrees with actual receive rows')
            expected = ref['recv_x']
            if expanded:
                exp = golden._expand_reference(ref['recv_x'], ref['recv_topk_idx'], ref['recv_topk_weights'], ref['expert_counts'])
                expected = exp['expanded_x']
                golden._assert_equal('expanded payload/scales', expansion[0], expected)
                eh = expansion[2]
                for name, value, wanted in [('offsets', eh.expert_offsets, exp['expert_offsets']), ('expert counts', eh.actual_expert_counts, exp['actual_expert_counts']), ('expand mapping', eh.recv_to_expanded_idx, exp['recv_to_expanded_idx'])]:
                    golden._assert_equal(name, value, wanted)
                if not cached:
                    golden._assert_equal('expanded weights', expansion[1], exp['expanded_topk_weights'])
                elif expansion[1] is not None:
                    raise AssertionError('payload-only cached expand returned weights')
            return golden._dequantize_x(expected, torch.bfloat16)
        fresh = wait(operations.dispatch(x))
        expert_ref = check_dispatch(fresh)
        dispatched, expansion = fresh[:2]
        handles = (dispatched[4], expansion[2] if expanded else None)
        recv = expansion[0] if expanded else dispatched[0]
        recv_weights = expansion[1] if expanded else dispatched[2]
        owned = golden._dequantize_x(recv, torch.bfloat16).clone()
        golden._assert_equal('expert input', owned, expert_ref)
        del expert_ref
        cached = wait(operations.dispatch(cached_x, handles))
        check_dispatch(cached, delta=1, cached=True)
        del cached
        bias0 = bias1 = bias = None
        if args.combine_bias == 'two':
            bias0 = torch.full((args.num_tokens, args.hidden), 0.5, dtype=torch.bfloat16)
            bias1 = (torch.arange(args.num_tokens * args.hidden, dtype=torch.float32).remainder_(2048).reshape(args.num_tokens, args.hidden) * 0.001).to(torch.bfloat16)
            bias = (bias0.to('npu'), bias1.to('npu'))

        def combine(weighted):
            return operations.combine(owned, handles, recv_weights if weighted else None, bias if weighted else None)
        source = golden._dequantize_x(golden._make_dispatch_x(rank, args), torch.bfloat16)
        for weighted in (False, True):
            result = wait(combine(weighted))[1]
            expected = torch.zeros_like(source, dtype=torch.float32)
            if weighted and bias is not None:
                expected += bias0.float()
                expected += bias1.float()
            for peer in range(args.ep_size):
                mask = (routes[rank] >= peer * args.experts_per_rank) & (routes[rank] < (peer + 1) * args.experts_per_rank)
                if expanded:
                    partial = torch.zeros_like(source, dtype=torch.float32)
                    for k in range(args.num_topk):
                        partial[mask[:, k]] += source[mask[:, k]].float()
                    expected += partial.to(torch.bfloat16).float()
                else:
                    rows = mask.any(1)
                    expected[rows] += source[rows].float()
            golden._assert_equal('combine payload', result[0], expected.to(torch.bfloat16))
            if weighted:
                expected_weights = golden._make_weights(rank, args).clone()
                expected_weights[routes[rank] == -1] = 0
                golden._assert_equal('combine weights', result[1], expected_weights)
            elif result[1] is not None:
                raise AssertionError('unweighted combine returned weights')
            del result, expected
        del source, bias0, bias1
        torch.npu.synchronize()
        write_json(destination / 'precision.json', dict(passed=True, fresh=True, cached=True, expanded=expanded, combine_unweighted=True, combine_weighted=True, combine_bias=args.combine_bias, combine_weighted_two_bias=args.combine_bias == 'two'))
        print(f'PRECISION_PASSED rank={rank} api=v1', flush=True)
        torch.npu.synchronize()
        check_dispatch(fresh, check_storage=False)
        write_json(destination / 'result.json', dict(rank=rank, device=device, precision_passed=True, destroyed=False, precision_only=True, operators={}, buffer_reused=True, buffer_identity=str(id(buffer))))
        write_json(destination / 'exit.json', dict(code=0, ended=time.time()))
        print(f'PRECISION_ONLY_PASSED rank={rank} api=v1', flush=True)
        return lambda: check_dispatch(fresh, check_storage=False)
    except BaseException:
        write_json(destination / 'exit.json', dict(code=1, ended=time.time(), traceback=traceback.format_exc()))
        raise
