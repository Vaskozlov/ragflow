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

from rag.nlp import append_context2table_image4pdf


def _section(text, page, top, bottom):
    return text, "text", [(page, 0, 100, top, bottom)]


def _media(page, top=30, bottom=70):
    return ((None, "IMAGE DESCRIPTION"), [(page, 20, 80, top, bottom)])


def test_media_context_does_not_cross_page_boundaries():
    sections = [
        _section("PREVIOUS PAGE", 0, 80, 90),
        _section("CURRENT PAGE ABOVE", 1, 10, 20),
        _section("CURRENT PAGE BELOW", 1, 80, 90),
        _section("NEXT PAGE", 2, 10, 20),
    ]

    result = append_context2table_image4pdf(sections, [_media(1)], table_context_size=100)
    content = result[0][0][1]

    assert content == "CURRENT PAGE ABOVE\nIMAGE DESCRIPTION\nCURRENT PAGE BELOW"
    assert "PREVIOUS PAGE" not in content
    assert "NEXT PAGE" not in content


def test_media_without_same_page_text_keeps_only_its_description():
    sections = [
        _section("PREVIOUS PAGE", 0, 80, 90),
        _section("NEXT PAGE", 2, 10, 20),
    ]

    result = append_context2table_image4pdf(sections, [_media(1)], table_context_size=100)

    assert result[0][0][1] == "IMAGE DESCRIPTION"
