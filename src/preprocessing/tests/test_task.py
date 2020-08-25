from preprocessing.task import parse_args
import json
import shlex


def test_parse_args():
    args = {
        'data_dir': 'path/to/data',
        'filters_dir': 'path/to/filters',
        'job_dir': 'path/to/output',
        'config': {
            'start': 1,
            'end': 2,
            'exclude': ['fast_decay', 'slow_decay']
        },
        'instruments': ['guitar', 'keyboard']
    }

    def jsonify(v):
        return v if isinstance(v, str) else json.dumps(v)
    # Test arg parsing for direct runner
    cmd = [[f'--{k}', jsonify(v)] for k, v in args.items()]
    cmd = [x for y in cmd for x in y]
    known_args, pipeline_args = parse_args(cmd)
    assert known_args == args
    assert pipeline_args == []
    # Test arg parsing for dataflow runner
    cmd.extend(['--runner', 'dataflow'])
    known_args, pipeline_args = parse_args(cmd)
    assert known_args == args
    dataflow_args = ['--runner', 'dataflow', '--project', '--region',
                    '--staging_location', '--temp_location', '--job_name']
    assert all([a.split('=')[0] in dataflow_args for a in pipeline_args])