import argparse
import json
import time

from .batch_queue import BatchQueue


def main():
    parser = argparse.ArgumentParser(description='Real-ESRGAN batch processing backend CLI')
    parser.add_argument('image_path', help='Input image path to process')
    parser.add_argument('--priority', choices=['high', 'normal', 'low'], default='normal')
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--poll-interval', type=float, default=0.5)
    parser.add_argument('--model-name', default='identity')
    parser.add_argument('--max-batch-size', type=int, default=1)
    args = parser.parse_args()

    queue = BatchQueue(model_name=args.model_name, max_batch_size=args.max_batch_size)
    queue.start()
    try:
        job_id = queue.submit_job(args.image_path, priority=args.priority, timeout=args.timeout)
        print(json.dumps({'job_id': job_id, 'state': 'pending'}))
        while True:
            status = queue.get_job_status(job_id)
            if status['state'] in {'completed', 'failed'}:
                payload = {'job_id': job_id, 'status': status}
                if status['state'] == 'completed':
                    result = queue.get_result(job_id)
                    output = result.pop('output', None)
                    if output is not None:
                        result['output_shape'] = list(output.shape)
                        result['output_dtype'] = str(output.dtype)
                    payload['result'] = result
                print(json.dumps(payload))
                break
            time.sleep(args.poll_interval)
    finally:
        queue.shutdown()


if __name__ == '__main__':
    main()
