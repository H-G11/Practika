"""Loopback-only image-upload viewer using actual YOLO inference, never canned predictions."""

import argparse
import os
from pathlib import Path

try:
    from .prediction import detection_report, predict_image
except ImportError:
    from prediction import detection_report, predict_image


def predict_upload(model, path, confidence, iou, *, device='cpu', imgsz=640):
    result = predict_image(model, path, confidence=confidence, iou=iou, device=device, imgsz=imgsz)
    # Ultralytics plot() is BGR; Gradio expects RGB.
    return result.plot()[:, :, ::-1], detection_report(result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--port', type=int, default=7861)
    args = parser.parse_args(argv)
    if not args.weights.is_file() or args.imgsz < 1 or not 1 <= args.port <= 65535:
        parser.error('Require existing trusted weights, imgsz > 0 and a valid local port')
    os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
    try:
        import gradio as gr
        from ultralytics import YOLO

        model = YOLO(str(args.weights.resolve()))
        with gr.Blocks(title='Waste detection | Local inference') as demo:
            gr.Markdown('# Waste detection\nLocal YOLO inference on your image. No prerecorded detections. Use authorized, non-sensitive images.')
            with gr.Row():
                source = gr.Image(type='filepath', label='Input image')
                output = gr.Image(label='Actual predictions', interactive=False)
            with gr.Row():
                confidence = gr.Slider(.01, 1, value=.25, step=.01, label='Confidence threshold')
                iou = gr.Slider(.01, 1, value=.7, step=.01, label='NMS IoU threshold')
            run = gr.Button('Run local inference', variant='primary')
            report = gr.JSON(label='Actual boxes and runtime')

            def infer(path, conf, threshold):
                try:
                    return predict_upload(model, path, conf, threshold, device=args.device, imgsz=args.imgsz)
                except (ValueError, OSError, RuntimeError) as exc:
                    raise gr.Error(str(exc)) from exc

            run.click(infer, [source, confidence, iou], [output, report], concurrency_limit=1)
        demo.queue(max_size=8).launch(server_name='127.0.0.1', server_port=args.port,
                                     share=False, inbrowser=False, show_error=False)
    except ImportError as exc:
        parser.exit(1, f'Missing demo dependency: {exc}. Install requirements-demo.txt\n')


if __name__ == '__main__':
    main()
