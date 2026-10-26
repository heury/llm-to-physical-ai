"""Publish only validated artifacts, using the locally cached Hugging Face login."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True)
    parser.add_argument('--repo', default='caspar/agent-rl-en')
    args = parser.parse_args()
    directory = Path(args.directory)
    report = json.loads((directory / 'validation.json').read_text())
    data = directory / 'agent_rl_en.jsonl'
    assert hashlib.sha256(data.read_bytes()).hexdigest() == report['output_sha256']
    assert sum(1 for _ in data.open()) == report['output_rows'] == 30196
    assert report['excluded_translate_text_rows'] == 9792
    api = HfApi()
    assert api.whoami()['name'] == args.repo.split('/')[0]
    if api.repo_exists(args.repo, repo_type='dataset'):
        raise RuntimeError('Destination already exists; inspect it before replacing files.')
    api.create_repo(args.repo, repo_type='dataset', private=False)
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(directory / name))
                  for name in ['agent_rl_en.jsonl', 'README.md', 'validation.json']]
    operations.append(CommitOperationAdd(path_in_repo='translate.py', path_or_fileobj=str(Path(__file__).with_name('translate.py'))))
    commit = api.create_commit(args.repo, repo_type='dataset', operations=operations,
                               commit_message='Add English MiniMind Agent RL, excluding translate_text records')
    info = api.dataset_info(args.repo)
    assert info.private is False
    downloaded = hf_hub_download(args.repo, 'agent_rl_en.jsonl', repo_type='dataset', revision=commit.oid,
                                 local_dir=str(directory / 'readback'))
    assert hashlib.sha256(Path(downloaded).read_bytes()).hexdigest() == report['output_sha256']
    result = {'url': 'https://huggingface.co/datasets/' + args.repo, 'commit': commit.oid,
              'public': not info.private, 'readback_sha256_verified': True}
    (directory / 'published.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
