# W.5 MoE smoke — BLOCKED

No MoE model from the preference list loaded successfully on this host.

## Preference list (all failed)

- Qwen/Qwen3.5-35B-A3B-Base
- mistralai/Mixtral-8x7B-Instruct-v0.1
- deepseek-ai/DeepSeek-V2-Lite-MoE

## Hardware

* device: mps
* torch: 2.11.0

## Error traces

### Qwen/Qwen3.5-35B-A3B-Base
```
Traceback (most recent call last):
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/transformers/utils/hub.py", line 422, in cached_files
    hf_hub_download(
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/huggingface_hub/utils/_validators.py", line 88, in _inner_fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/huggingface_hub/file_download.py", line 997, in hf_hub_download
    return _hf_hub_download_to_cache_dir(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/huggingface_hub/file_download.py", line 1148, in _hf_hub_download_to_cache_dir
    _raise_on_head_call_error(head_call_error, force_download, local_files_only)
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/huggingface_hub/file_download.py", line 1773, in _raise_on_head_call_error
    raise LocalEntryNotFoundError(
huggingface_hub.errors.LocalEntryNotFoundError: Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable hf.co look-ups and downloads online, set 'local_files_only' to False.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/transformers/models/auto/tokenization_auto.py", line 687, in from_pretrained
    config = AutoConfig.from_pretrained(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/transformers/models/auto/configuration_auto.py", line 374, in from_pretrained
    config_dict, unused_kwargs = PreTrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/transformers/configuration_utils.py", line 680, in get_config_dict
    config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/transformers/configuration_utils.py", line 735, in _get_config_dict
    resolved_config_file = cached_file(
                           ^^^^^^^^^^^^
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/transformers/utils/hub.py", line 278, in cached_file
    file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/transformers/utils/hub.py", line 498, in cached_files
    raise OSError(
OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/transformers/utils/hub.py", line 422, in cached_files
    hf_hub_download(
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/huggingface_hub/utils/_validators.py", line 88, in _inner_fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/huggingface_hub/file_download.py", line 997, in hf_hub_download
    return _hf_hub_download_to_cache_dir(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/huggingface_hub/file_download.py", line 1148, in _hf_hub_download_to_cache_dir
    _raise_on_head_call_error(head_call_error, force_download, local_files_only)
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/huggingface_hub/file_download.py", line 1773, in _raise_on_head_call_error
    raise LocalEntryNotFoundError(
huggingface_hub.errors.LocalEntryNotFoundError: Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable hf.co look-ups and downloads online, set 'local_files_only' to False.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/gabiri/projects/RCV-HC/experiments/W5_moe/run.py", line 143, in try_load_moe_model
    tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/transformers/models/auto/tokenization_auto.py", line 691, in from_pretrained
    config = PreTrainedConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/transformers/configuration_utils.py", line 639, in from_pretrained
    config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/transformers/configuration_utils.py", line 680, in get_config_dict
    config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/transformers/configuration_utils.py", line 735, in _get_config_dict
    resolved_config_file = cached_file(
                           ^^^^^^^^^^^^
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/transformers/utils/hub.py", line 278, in cached_file
    file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/transformers/utils/hub.py", line 498, in cached_files
    raise OSError(
OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.

```

### mistralai/Mixtral-8x7B-Instruct-v0.1
```
Traceback (most recent call last):
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/transformers/utils/hub.py", line 422, in cached_files
    hf_hub_download(
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/huggingface_hub/utils/_validators.py", line 88, in _inner_fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/huggingface_hub/file_download.py", line 997, in hf_hub_download
    return _hf_hub_download_to_cache_dir(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/huggingface_hub/file_download.py", line 1148, in _hf_hub_download_to_cache_dir
    _raise_on_head_call_error(head_call_error, force_download, local_files_only)
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/huggingface_hub/file_download.py", line 1773, in _raise_on_head_call_error
    raise LocalEntryNotFoundError(
huggingface_hub.errors.LocalEntryNotFoundError: Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable hf.co look-ups and downloads online, set 'local_files_only' to False.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/transformers/models/auto/tokenization_auto.py", line 687, in from_pretrained
    config = AutoConfig.from_pretrained(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/transformers/models/auto/configuration_auto.py", line 374, in from_pretrained
    config_dict, unused_kwargs = PreTrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/transformers/configuration_utils.py", line 680, in get_config_dict
    config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/transformers/configuration_utils.py", line 735, in _get_config_dict
    resolved_config_file = cached_file(
                           ^^^^^^^^^^^^
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/transformers/utils/hub.py", line 278, in cached_file
    file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/transformers/utils/hub.py", line 498, in cached_files
    raise OSError(
OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/transformers/utils/hub.py", line 422, in cached_files
    hf_hub_download(
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/huggingface_hub/utils/_validators.py", line 88, in _inner_fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/huggingface_hub/file_download.py", line 997, in hf_hub_download
    return _hf_hub_download_to_cache_dir(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/huggingface_hub/file_download.py", line 1148, in _hf_hub_download_to_cache_dir
    _raise_on_head_call_error(head_call_error, force_download, local_files_only)
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/huggingface_hub/file_download.py", line 1773, in _raise_on_head_call_error
    raise LocalEntryNotFoundError(
huggingface_hub.errors.LocalEntryNotFoundError: Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable hf.co look-ups and downloads online, set 'local_files_only' to False.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/gabiri/projects/RCV-HC/experiments/W5_moe/run.py", line 143, in try_load_moe_model
    tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/transformers/models/auto/tokenization_auto.py", line 691, in from_pretrained
    config = PreTrainedConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/transformers/configuration_utils.py", line 639, in from_pretrained
    config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/transformers/configuration_utils.py", line 680, in get_config_dict
    config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/transformers/configuration_utils.py", line 735, in _get_config_dict
    resolved_config_file = cached_file(
                           ^^^^^^^^^^^^
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/transformers/utils/hub.py", line 278, in cached_file
    file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/transformers/utils/hub.py", line 498, in cached_files
    raise OSError(
OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.

```

### deepseek-ai/DeepSeek-V2-Lite-MoE
```
Traceback (most recent call last):
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/transformers/utils/hub.py", line 422, in cached_files
    hf_hub_download(
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/huggingface_hub/utils/_validators.py", line 88, in _inner_fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/huggingface_hub/file_download.py", line 997, in hf_hub_download
    return _hf_hub_download_to_cache_dir(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/huggingface_hub/file_download.py", line 1148, in _hf_hub_download_to_cache_dir
    _raise_on_head_call_error(head_call_error, force_download, local_files_only)
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/huggingface_hub/file_download.py", line 1773, in _raise_on_head_call_error
    raise LocalEntryNotFoundError(
huggingface_hub.errors.LocalEntryNotFoundError: Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable hf.co look-ups and downloads online, set 'local_files_only' to False.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/transformers/models/auto/tokenization_auto.py", line 687, in from_pretrained
    config = AutoConfig.from_pretrained(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/transformers/models/auto/configuration_auto.py", line 374, in from_pretrained
    config_dict, unused_kwargs = PreTrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/transformers/configuration_utils.py", line 680, in get_config_dict
    config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/transformers/configuration_utils.py", line 735, in _get_config_dict
    resolved_config_file = cached_file(
                           ^^^^^^^^^^^^
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/transformers/utils/hub.py", line 278, in cached_file
    file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/transformers/utils/hub.py", line 498, in cached_files
    raise OSError(
OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/transformers/utils/hub.py", line 422, in cached_files
    hf_hub_download(
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/huggingface_hub/utils/_validators.py", line 88, in _inner_fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/huggingface_hub/file_download.py", line 997, in hf_hub_download
    return _hf_hub_download_to_cache_dir(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/huggingface_hub/file_download.py", line 1148, in _hf_hub_download_to_cache_dir
    _raise_on_head_call_error(head_call_error, force_download, local_files_only)
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/huggingface_hub/file_download.py", line 1773, in _raise_on_head_call_error
    raise LocalEntryNotFoundError(
huggingface_hub.errors.LocalEntryNotFoundError: Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable hf.co look-ups and downloads online, set 'local_files_only' to False.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/gabiri/projects/RCV-HC/experiments/W5_moe/run.py", line 143, in try_load_moe_model
    tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/transformers/models/auto/tokenization_auto.py", line 691, in from_pretrained
    config = PreTrainedConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/transformers/configuration_utils.py", line 639, in from_pretrained
    config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/transformers/configuration_utils.py", line 680, in get_config_dict
    config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/transformers/configuration_utils.py", line 735, in _get_config_dict
    resolved_config_file = cached_file(
                           ^^^^^^^^^^^^
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/transformers/utils/hub.py", line 278, in cached_file
    file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/gabiri/projects/RCV-HC/.venv-mac/lib/python3.11/site-packages/transformers/utils/hub.py", line 498, in cached_files
    raise OSError(
OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.

```


## Resolution

This is the expected outcome on the 64 GB development machine: none of
the listed MoE checkpoints are downloaded locally and the runner is
configured with ``local_files_only=True`` to avoid an unintended
multi-GB download.

The full W.5 grid (1890 cells) will run on the 128 GB GB10 host; per
``plan.md`` G-10 the W.5 smoke is allowed to be deferred to that
environment.  The patcher implementation, unit tests, and aggregator
are validated locally via the synthetic mock adapter — see
``tests/test_moe_attn_patcher.py``.
