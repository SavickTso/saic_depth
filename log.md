Traceback (most recent call last):
  File "/root/saic_depth/tools/test_net.py", line 125, in <module>
    main()
  File "/root/saic_depth/tools/test_net.py", line 115, in main
    inference(
  File "/usr/local/lib/python3.10/dist-packages/saic_depth_completion-0.0.1-py3.10.egg/saic_depth_completion/engine/inference.py", line 28, in inference
  File "/usr/local/lib/python3.10/dist-packages/tqdm/std.py", line 1182, in __iter__
    for obj in iterable:
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/dataloader.py", line 630, in __next__
    data = self._next_data()
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/dataloader.py", line 1345, in _next_data
    return self._process_data(data)
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/dataloader.py", line 1371, in _process_data
    data.reraise()
  File "/usr/local/lib/python3.10/dist-packages/torch/_utils.py", line 694, in reraise
    raise exception
TypeError: Caught TypeError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/_utils/worker.py", line 308, in _worker_loop
    data = fetcher.fetch(index)
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/_utils/fetch.py", line 54, in fetch
    return self.collate_fn(data)
  File "/usr/local/lib/python3.10/dist-packages/saic_depth_completion-0.0.1-py3.10.egg/saic_depth_completion/data/collate.py", line 13, in default_collate
    batch[k] = torch.stack(v)
TypeError: expected Tensor as element 0 in argument 0, but got str
