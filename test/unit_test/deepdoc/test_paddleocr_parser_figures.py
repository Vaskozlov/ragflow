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
