from bench_v2_precision import *

def check_case(args, buffer):
    assert args.precision_only, 'Batch checker must not run performance'
    sys.path.insert(0, str(REPO / 'tests/clos'))
    import torch
    import torch.distributed as dist
    import torch_npu
    from deep_ep.backends.global_clos import ElasticBuffer
    from deep_ep import topk_idx_t
    import common as golden
    import profiling
    rank = int(os.environ['RANK'])
    if int(os.environ['WORLD_SIZE']) != args.ep_size:
        raise RuntimeError('torchrun world size does not match EP size')
    local_rank = int(os.environ['LOCAL_RANK'])
    if int(os.environ['LOCAL_WORLD_SIZE']) != local_world_size(args) or not 0 <= local_rank < local_world_size(args) or rank != args.node_rank * local_world_size(args) + local_rank:
        raise RuntimeError('torchrun local/global rank mapping does not match this node')
    args.num_aiv, args.dtype, expanded = (args.aiv[0], args.dtypes[0], bool(args.expand[0]))
    args.expert_alignment = 1
    args.profile_warmup, args.profile_iters = (args.warmup, args.iterations)
    args.profile_sample_markers = True
    args.combine_notify = True
    args.all_rank_sync_each_iteration = True
    os.environ[COMBINE_NOTIFY_ENV] = '0'
    device = args.first_device + local_rank
    destination = args.output / f'rank{rank}'
    destination.mkdir(parents=True, exist_ok=False)
    started = time.time()
    write_json(destination / 'process.json', dict(pid=os.getpid(), rank=rank, device=device, node_rank=args.node_rank, started=started))
    try:
        torch.set_num_threads(1)
        routes = [golden._make_topk_idx(p, args) for p in range(args.ep_size)]
        counts = route_counts([r.tolist() for r in routes], args.experts_per_rank)
        write_json(destination / 'route_load.json', route_statistics([r.tolist() for r in routes], args.experts_per_rank, args.route_group_size))
        write_json(destination / 'input_routes.json', routes[rank].tolist())
        torch.npu.set_device(device)
        args.effective_udma_config = effective_udma_config(buffer, num_aiv=args.num_aiv, row_bytes=args.hidden * (1 if args.dtype == 'fp8' else 2), bf16=args.dtype == 'bf16', expanded=expanded, combine_row_bytes=args.hidden * 2)
        if rank == 0:
            print('EFFECTIVE_UDMA_CONFIG', json.dumps(args.effective_udma_config, sort_keys=True), flush=True)
        if getattr(args, 'udma_auto_policy', False):
            args.resolved_udma_policy = args.effective_udma_config['operation_plans']['dispatch']
            if rank == 0:
                print('AUTO_POLICY', json.dumps(args.resolved_udma_policy, sort_keys=True), flush=True)
        identity = loaded_identity()
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
        dispatch_ledger = byte_ledger(counts, rank, tokens=args.num_tokens, hidden=args.hidden, topk=args.num_topk, dtype=args.dtype, idx_bytes=idx.element_size(), op='dispatch')
        if input_bytes != args.num_tokens * dispatch_ledger['official_logical_bytes_per_row']:
            raise ValueError('logical row-byte formula does not match actual dispatch input tensors')

        def wait(result):
            profiling._consume_profile_result(result)
            return result

        def dispatch(value, handle=None):
            return buffer.dispatch(value, topk_idx=idx if handle is None else None, topk_weights=weights if handle is None else None, handle=handle, num_experts=args.ep_size * args.experts_per_rank, num_sms=args.num_aiv, do_expand=expanded, expert_alignment=1, async_with_compute_stream=True)

        def check_dispatch(result, delta=0, cached=False):
            recv, recv_idx, recv_weights, handle, _ = result
            ref = golden._dispatch_reference(rank, args, delta=delta)
            if expanded:
                exp = golden._expand_reference(ref['recv_x'], ref['recv_topk_idx'], ref['recv_topk_weights'], ref['expert_counts'])
                expected = exp['expanded_x']
                expected_weights = exp['expanded_topk_weights']
                if recv_idx is not None:
                    raise AssertionError('expanded output unexpectedly has rank topk indices')
                offsets, actual, mapping, aligned = handle._expand_layout
                for name, value, wanted in [('offsets', offsets, exp['expert_offsets']), ('expert counts', actual, exp['actual_expert_counts']), ('expand mapping', mapping, exp['recv_to_expanded_idx'])]:
                    golden._assert_equal(name, value, wanted)
                if aligned != ref['expert_counts'].tolist():
                    raise AssertionError('aligned expert counts mismatch')
            else:
                expected, expected_weights = (ref['recv_x'], ref['recv_topk_weights'])
                if not cached:
                    golden._assert_equal('dispatch indices', recv_idx, ref['recv_topk_idx'])
                elif recv_idx is not None:
                    raise AssertionError('payload-only cached dispatch returned topk')
            golden._assert_equal('dispatch payload/scales', recv, expected)
            if not cached:
                golden._assert_equal('dispatch weights', recv_weights, expected_weights)
            elif recv_weights is not None:
                raise AssertionError('payload-only cached dispatch returned weights')
            for name, value, wanted in [('src_idx', handle._dispatch_layout[3], ref['recv_src_idx']), ('send_head', handle._dispatch_layout[5], ref['send_head']), ('recv prefix', handle._dispatch_layout[2], ref['recv_channel_prefix'])]:
                golden._assert_equal(name, value, wanted)
            expected_rows = sum((counts['rows'][p][rank] for p in range(args.ep_size)))
            if handle.num_recv_tokens != expected_rows:
                raise AssertionError('independent route count disagrees with actual receive rows')
            return golden._dequantize_x(expected, torch.bfloat16)
        fresh = wait(dispatch(x))
        expert_ref = check_dispatch(fresh)
        recv, _, recv_weights, handle, _ = fresh
        owned = golden._dequantize_x(recv, torch.bfloat16).clone()
        golden._assert_equal('expert input', owned, expert_ref)
        del expert_ref
        cached = wait(dispatch(cached_x, handle))
        check_dispatch(cached, delta=1, cached=True)
        del cached
        bias0 = bias1 = bias = None
        if args.combine_bias == 'two':
            bias0 = torch.full((args.num_tokens, args.hidden), 0.5, dtype=torch.bfloat16)
            bias1 = (torch.arange(args.num_tokens * args.hidden, dtype=torch.float32).remainder_(2048).reshape(args.num_tokens, args.hidden) * 0.001).to(torch.bfloat16)
            bias = (bias0.to('npu'), bias1.to('npu'))

        def combine(weighted):
            return buffer.combine(owned, handle, topk_weights=recv_weights if weighted else None, bias=bias if weighted else None, num_sms=args.num_aiv, async_with_compute_stream=True)
        source = golden._dequantize_x(golden._make_dispatch_x(rank, args), torch.bfloat16)
        for weighted in (False, True):
            result = wait(combine(weighted))
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
        write_json(destination / 'precision.json', dict(passed=True, fresh=True, cached=True, combine_unweighted=True, combine_weighted=True, combine_bias=args.combine_bias, combine_weighted_two_bias=args.combine_bias == 'two'))
        print(f'PRECISION_PASSED rank={rank}', flush=True)
        torch.npu.synchronize()
        write_json(destination / 'result.json', dict(rank=rank, device=device, precision_passed=True, destroyed=False, precision_only=True, operators={}, buffer_reused=True, buffer_identity=str(id(buffer))))
        write_json(destination / 'exit.json', dict(code=0, ended=time.time()))
        print(f'PRECISION_ONLY_PASSED rank={rank}', flush=True)
        return None
    except BaseException:
        write_json(destination / 'exit.json', dict(code=1, ended=time.time(), traceback=traceback.format_exc()))
        raise
