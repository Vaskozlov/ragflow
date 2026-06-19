#
#  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import base64
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from unittest import skipUnless
from unittest.mock import Mock, patch

from PIL import Image

from deepdoc.parser.paddleocr_parser import PaddleOCRConfig, PaddleOCRParser


def _result_with_blocks(*blocks):
    return {
        "layoutParsingResults": [
            {
                "prunedResult": {
                    "parsing_res_list": list(blocks),
                }
            }
        ]
    }


def test_paddleocr_vl_16_forwards_image_block_option():
    parser = PaddleOCRParser(base_url="http://paddleocr.test", access_token="dummy", algorithm="PaddleOCR-VL-1.6")
    config = PaddleOCRConfig.from_dict(
        {
            "algorithm": "PaddleOCR-VL-1.6",
            "algorithm_config": {"use_ocr_for_image_block": True},
        }
    )

    payload = parser._build_payload(b"", 0, config)

    assert payload["useOcrForImageBlock"] is True


def test_figure_blocks_are_cropped_and_excluded_from_text_sections():
    parser = PaddleOCRParser(base_url="http://paddleocr.test", access_token="dummy")
    parser.page_images = [Image.new("RGB", (100, 100), "white")]
    parser.page_from = 3
    result = _result_with_blocks(
        {
            "block_label": "figure",
            "block_content": '<img src="figure.png"/>',
            "block_bbox": [20, 40, 140, 160],
        },
        {
            "block_label": "figure_caption",
            "block_content": "An existing figure caption",
            "block_bbox": [20, 162, 140, 180],
        },
        {
            "block_label": "text",
            "block_content": "Body text",
            "block_bbox": [10, 10, 80, 30],
        },
    )

    sections = parser._transfer_to_sections(result, algorithm="PaddleOCR-VL", parse_method="manual")
    figures = parser._transfer_to_tables(result)

    assert [section[0] for section in sections] == ["An existing figure caption", "Body text"]
    assert len(figures) == 1
    (image, descriptions), positions = figures[0]
    assert image.size == (60, 60)
    assert descriptions == [""]
    assert positions == [(3, 10, 70, 20, 80)]


def test_non_figure_blocks_do_not_create_image_items():
    parser = PaddleOCRParser(base_url="http://paddleocr.test", access_token="dummy")
    parser.page_images = [Image.new("RGB", (100, 100), "white")]
    parser.page_from = 0
    result = _result_with_blocks(
        {
            "block_label": "table",
            "block_content": "<table></table>",
            "block_bbox": [0, 0, 100, 100],
        },
        {
            "block_label": "figure_title",
            "block_content": "Figure 1",
            "block_bbox": [0, 0, 100, 20],
        },
    )

    assert parser._transfer_to_tables(result) == []


def test_pdfium_failure_uses_ghostscript_fallback():
    parser = PaddleOCRParser(base_url="http://paddleocr.test", access_token="dummy")
    rendered_pages = [Image.new("RGB", (100, 100), "white")]

    try:
        with (
            patch("deepdoc.parser.paddleocr_parser.pdfplumber.open", side_effect=RuntimeError("PDFium failed")),
            patch.object(parser, "_render_pages_with_ghostscript", return_value=rendered_pages) as render_fallback,
        ):
            parser.__images__(b"%PDF-test", page_from=0, page_to=1)

        render_fallback.assert_called_once_with(b"%PDF-test", 0, 1)
        assert parser.page_images == rendered_pages
    finally:
        for image in rendered_pages:
            image.close()


def test_missing_page_images_warn_only_once_during_chunk_cropping():
    parser = PaddleOCRParser(base_url="http://paddleocr.test", access_token="dummy")
    parser.page_images = []
    parser.logger = Mock()
    position = "@@1\t0\t10\t0\t10##"

    assert parser.crop(position, need_position=True) == (None, None)
    assert parser.crop(position, need_position=True) == (None, None)

    parser.logger.warning.assert_called_once()


@skipUnless(shutil.which("gs"), "Ghostscript is not installed")
def test_ghostscript_renders_pdf_bytes_when_available():
    parser = PaddleOCRParser(base_url="http://paddleocr.test", access_token="dummy")
    test_pdf = Path(__file__).resolve().parents[2] / "benchmark/test_docs/Doc1.pdf"

    images = parser._render_pages_with_ghostscript(test_pdf.read_bytes(), page_from=0, page_to=1)

    try:
        assert len(images) == 1
        assert images[0].width > 0
        assert images[0].height > 0
    finally:
        for image in images:
            image.close()


def test_pdfium_rendering_is_serialized_across_parser_instances():
    active_calls = 0
    max_active_calls = 0
    state_lock = threading.Lock()
    start_barrier = threading.Barrier(2)

    class EmptyPdf:
        pages = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def open_pdf(_source):
        nonlocal active_calls, max_active_calls
        with state_lock:
            active_calls += 1
            max_active_calls = max(max_active_calls, active_calls)
        time.sleep(0.05)
        with state_lock:
            active_calls -= 1
        return EmptyPdf()

    def render(parser):
        start_barrier.wait()
        parser.__images__(b"%PDF-test", page_from=0, page_to=1)

    parsers = [
        PaddleOCRParser(base_url="http://paddleocr.test", access_token="dummy"),
        PaddleOCRParser(base_url="http://paddleocr.test", access_token="dummy"),
    ]
    with patch("deepdoc.parser.paddleocr_parser.pdfplumber.open", side_effect=open_pdf):
        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(render, parsers))

    assert max_active_calls == 1


def test_paddle_result_images_are_used_when_local_renderers_fail():
    parser = PaddleOCRParser(base_url="http://paddleocr.test", access_token="dummy")
    parser.page_images = []
    source_image = Image.new("RGB", (200, 100), "white")
    encoded = BytesIO()
    source_image.save(encoded, format="JPEG")
    source_image.close()
    result = {
        "layoutParsingResults": [
            {
                "inputImage": base64.b64encode(encoded.getvalue()).decode("ascii"),
                "prunedResult": {
                    "parsing_res_list": [
                        {
                            "block_label": "image",
                            "block_content": '<img src="figure.jpg"/>',
                            "block_bbox": [20, 20, 100, 80],
                        }
                    ]
                },
            }
        ]
    }

    figures = parser._transfer_to_tables(result)

    try:
        assert len(parser.page_images) == 1
        assert parser.page_images[0].size == (100, 50)
        assert len(figures) == 1
        (figure, descriptions), positions = figures[0]
        assert figure.size == (40, 30)
        assert descriptions == [""]
        assert positions == [(0, 10, 50, 10, 40)]
    finally:
        encoded.close()
        for image in parser.page_images:
            image.close()


def test_both_local_renderers_can_fail_without_aborting_paddle_text():
    parser = PaddleOCRParser(base_url="http://paddleocr.test", access_token="dummy")
    result = _result_with_blocks(
        {
            "block_label": "text",
            "block_content": "PaddleOCR text survives",
            "block_bbox": [0, 0, 100, 20],
        }
    )

    with (
        patch("deepdoc.parser.paddleocr_parser.pdfplumber.open", side_effect=RuntimeError("PDFium failed")),
        patch.object(parser, "_render_pages_with_ghostscript", side_effect=RuntimeError("Ghostscript failed")),
        patch.object(parser, "_send_request", return_value=result),
    ):
        sections, figures = parser.parse_pdf("ignored.pdf", binary=b"%PDF-test")

    assert sections == [("PaddleOCR text survives", "@@1\t0.0\t50.0\t0.0\t10.0##")]
    assert figures == []
    assert parser.page_images == []
