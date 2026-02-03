"""Unit tests for OCR metrics calculation functions

Tests edge cases and boundary conditions for:
- Edit distance calculation
- Normalized edit distance
- Edit distance similarity
- Empty string handling
- Length difference scenarios
"""

import pytest

from onnxtools.utils.ocr_metrics import calculate_edit_distance_metrics


class TestEditDistanceEdgeCases:
    """Unit tests for edit distance edge cases"""

    def test_edit_distance_empty_strings(self):
        """Test empty string boundary case"""
        dist, norm_ed, sim = calculate_edit_distance_metrics("", "")

        assert dist == 0
        assert norm_ed == 0.0
        assert sim == 1.0

    def test_edit_distance_one_empty_string(self):
        """Test one empty string case"""
        # Empty predicted, non-empty ground truth
        dist1, norm_ed1, sim1 = calculate_edit_distance_metrics("", "京A12345")
        assert dist1 == 7
        assert norm_ed1 == 1.0
        assert sim1 == 0.0

        # Non-empty predicted, empty ground truth
        dist2, norm_ed2, sim2 = calculate_edit_distance_metrics("京A12345", "")
        assert dist2 == 7
        assert norm_ed2 == 1.0
        assert sim2 == 0.0

    def test_edit_distance_identical_strings(self):
        """Test identical strings"""
        dist, norm_ed, sim = calculate_edit_distance_metrics("京A12345", "京A12345")

        assert dist == 0
        assert norm_ed == 0.0
        assert sim == 1.0

    def test_edit_distance_completely_different(self):
        """Test completely different strings of same length"""
        dist, norm_ed, sim = calculate_edit_distance_metrics("京A12345", "沪B67890")

        # All 7 characters are different
        assert dist == 7
        assert norm_ed == 1.0
        assert sim == 0.0

    def test_edit_distance_length_difference(self):
        """Test length difference scenarios"""
        # Shorter predicted
        dist, norm_ed, sim = calculate_edit_distance_metrics("京A123", "京A12345")
        assert dist == 2  # Missing "45"
        assert abs(norm_ed - 0.286) < 0.01  # 2/7 ≈ 0.286
        assert abs(sim - 0.714) < 0.01      # 1 - 0.286 ≈ 0.714

        # Longer predicted
        dist2, norm_ed2, sim2 = calculate_edit_distance_metrics("京A12345XY", "京A12345")
        assert dist2 == 2  # Extra "XY"
        assert abs(norm_ed2 - 0.222) < 0.01  # 2/9 ≈ 0.222
        assert abs(sim2 - 0.778) < 0.01      # 1 - 0.222 ≈ 0.778

    def test_edit_distance_single_character(self):
        """Test single character strings"""
        # Same character
        dist1, norm_ed1, sim1 = calculate_edit_distance_metrics("京", "京")
        assert dist1 == 0
        assert norm_ed1 == 0.0
        assert sim1 == 1.0

        # Different characters
        dist2, norm_ed2, sim2 = calculate_edit_distance_metrics("京", "沪")
        assert dist2 == 1
        assert norm_ed2 == 1.0
        assert sim2 == 0.0

    def test_edit_distance_insertion(self):
        """Test insertion operations"""
        dist, norm_ed, sim = calculate_edit_distance_metrics("京12345", "京A12345")

        # One character inserted (A)
        assert dist == 1
        assert abs(norm_ed - 0.143) < 0.01  # 1/7 ≈ 0.143
        assert abs(sim - 0.857) < 0.01      # 1 - 0.143 ≈ 0.857

    def test_edit_distance_deletion(self):
        """Test deletion operations"""
        dist, norm_ed, sim = calculate_edit_distance_metrics("京A12345", "京12345")

        # One character deleted (A)
        assert dist == 1
        assert abs(norm_ed - 0.143) < 0.01  # 1/7 ≈ 0.143
        assert abs(sim - 0.857) < 0.01

    def test_edit_distance_substitution(self):
        """Test substitution operations"""
        dist, norm_ed, sim = calculate_edit_distance_metrics("京A12345", "京B12345")

        # One character substituted (A -> B)
        assert dist == 1
        assert abs(norm_ed - 0.143) < 0.01
        assert abs(sim - 0.857) < 0.01

    def test_edit_distance_transposition(self):
        """Test character transposition"""
        dist, norm_ed, sim = calculate_edit_distance_metrics("京A21345", "京A12345")

        # Two characters swapped (2 and 1)
        # Levenshtein distance counts this as 2 operations (delete+insert)
        assert dist == 2
        assert abs(norm_ed - 0.286) < 0.01
        assert abs(sim - 0.714) < 0.01

    def test_edit_distance_chinese_characters(self):
        """Test with various Chinese characters"""
        dist, norm_ed, sim = calculate_edit_distance_metrics("京沪粤苏浙", "京沪渝川贵")

        # First 2 characters match, last 3 are different
        assert dist == 3
        assert abs(norm_ed - 0.6) < 0.01  # 3/5 = 0.6
        assert abs(sim - 0.4) < 0.01      # 1 - 0.6 = 0.4

    def test_edit_distance_numbers_only(self):
        """Test with number-only strings"""
        dist, norm_ed, sim = calculate_edit_distance_metrics("12345", "12346")

        # Last digit different
        assert dist == 1
        assert abs(norm_ed - 0.2) < 0.01  # 1/5 = 0.2
        assert abs(sim - 0.8) < 0.01      # 1 - 0.2 = 0.8

    def test_edit_distance_mixed_content(self):
        """Test with mixed Chinese, letters, and numbers"""
        dist1, norm_ed1, sim1 = calculate_edit_distance_metrics("京A1B2C3", "京A1B2C3")
        assert dist1 == 0
        assert norm_ed1 == 0.0
        assert sim1 == 1.0

        dist2, norm_ed2, sim2 = calculate_edit_distance_metrics("京A1B2C3", "京A1B2D4")
        # Last 2 characters different (C3 vs D4)
        assert dist2 == 2
        assert abs(norm_ed2 - 0.286) < 0.01
        assert abs(sim2 - 0.714) < 0.01

    def test_edit_distance_whitespace_handling(self):
        """Test whitespace handling"""
        # Leading/trailing spaces
        dist1, norm_ed1, sim1 = calculate_edit_distance_metrics(" 京A12345 ", "京A12345")
        assert dist1 == 2  # 2 spaces
        assert abs(norm_ed1 - 0.222) < 0.01  # 2/9

        # Internal spaces
        dist2, norm_ed2, sim2 = calculate_edit_distance_metrics("京 A 12345", "京A12345")
        assert dist2 == 2  # 2 internal spaces
        assert abs(norm_ed2 - 0.222) < 0.01

    def test_edit_distance_special_characters(self):
        """Test special characters in plate text"""
        # Plates with special markers (e.g., police plates)
        dist, norm_ed, sim = calculate_edit_distance_metrics("京A·12345", "京A-12345")

        # One character different (· vs -)
        assert dist == 1
        assert abs(norm_ed - 0.125) < 0.01  # 1/8

    def test_edit_distance_case_sensitivity(self):
        """Test case sensitivity (should be case-sensitive)"""
        dist, norm_ed, sim = calculate_edit_distance_metrics("京A12345", "京a12345")

        # 'A' and 'a' are different characters
        assert dist == 1
        assert abs(norm_ed - 0.143) < 0.01

    def test_edit_distance_normalized_range(self):
        """Test normalized edit distance is always in [0, 1]"""
        test_cases = [
            ("", ""),
            ("京", "沪"),
            ("京A12345", "京A12345"),
            ("京A12345", "沪B67890"),
            ("京A", "京A12345"),
            ("京A12345678", "京A"),
        ]

        for pred, gt in test_cases:
            _, norm_ed, sim = calculate_edit_distance_metrics(pred, gt)
            assert 0.0 <= norm_ed <= 1.0, f"norm_ed out of range for ({pred}, {gt})"
            assert 0.0 <= sim <= 1.0, f"sim out of range for ({pred}, {gt})"
            # Verify relationship
            assert abs(norm_ed + sim - 1.0) < 0.001, f"norm_ed + sim != 1.0 for ({pred}, {gt})"

    def test_edit_distance_symmetry(self):
        """Test that distance is symmetric but normalized distance is not"""
        pred = "京A12345"
        gt = "京A123"

        # Distance should be symmetric
        dist1, norm_ed1, sim1 = calculate_edit_distance_metrics(pred, gt)
        dist2, norm_ed2, sim2 = calculate_edit_distance_metrics(gt, pred)

        assert dist1 == dist2  # Raw distance is symmetric
        # Normalized distance uses max length, so it's the same
        assert abs(norm_ed1 - norm_ed2) < 0.001
        assert abs(sim1 - sim2) < 0.001

    def test_edit_distance_very_long_strings(self):
        """Test with very long plate texts (edge case)"""
        long_pred = "京A" + "1" * 100
        long_gt = "京A" + "1" * 100

        dist, norm_ed, sim = calculate_edit_distance_metrics(long_pred, long_gt)
        assert dist == 0
        assert norm_ed == 0.0
        assert sim == 1.0

        # One character different
        long_pred_diff = "京A" + "1" * 99 + "2"
        dist2, norm_ed2, sim2 = calculate_edit_distance_metrics(long_pred_diff, long_gt)
        assert dist2 == 1
        assert abs(norm_ed2 - 1/102) < 0.001  # 1/102


class TestEditDistanceRealWorldScenarios:
    """Unit tests for real-world OCR scenarios"""

    def test_ocr_common_confusion(self):
        """Test common OCR character confusions"""
        # 0 vs O
        dist1, norm_ed1, sim1 = calculate_edit_distance_metrics("京AO1234", "京A01234")
        assert dist1 == 1

        # 8 vs B
        dist2, norm_ed2, sim2 = calculate_edit_distance_metrics("京AB1234", "京A81234")
        assert dist2 == 1

        # 5 vs S
        dist3, norm_ed3, sim3 = calculate_edit_distance_metrics("京AS1234", "京A51234")
        assert dist3 == 1

    def test_ocr_partial_recognition(self):
        """Test partial recognition scenarios"""
        # Only first few characters recognized
        dist, norm_ed, sim = calculate_edit_distance_metrics("京A12", "京A12345")
        assert dist == 3
        assert abs(norm_ed - 0.429) < 0.01  # 3/7

    def test_ocr_double_layer_plate(self):
        """Test double-layer plate scenarios"""
        # Missing newline in prediction
        dist, norm_ed, sim = calculate_edit_distance_metrics("京A12345D", "京A12345\nD")
        assert dist == 1  # Missing newline character

    def test_ocr_noise_artifacts(self):
        """Test with noise artifacts"""
        # Extra dot/noise recognized
        dist, norm_ed, sim = calculate_edit_distance_metrics("京.A12345", "京A12345")
        assert dist == 1
        assert abs(norm_ed - 0.125) < 0.01  # 1/8
