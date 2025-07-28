import json
import tempfile
from pathlib import Path
from typing import Any

import PIL.Image
import pytest

import pixeltable as pxt
import pixeltable.exceptions as excs


class TestImportCoco:
    def create_sample_coco_data(self) -> dict[str, Any]:
        """Create a sample COCO annotation dataset for testing."""
        return {
            'info': {
                'description': 'Test COCO Dataset',
                'version': '1.0',
                'year': 2024,
                'contributor': 'Test',
                'date_created': '2024-01-01 00:00:00',
            },
            'licenses': [{'id': 1, 'name': 'Test License', 'url': 'https://example.com/license'}],
            'images': [
                {
                    'id': 1,
                    'width': 100,
                    'height': 100,
                    'file_name': 'image1.jpg',
                    'license': 1,
                    'date_captured': '2024-01-01 10:00:00',
                },
                {
                    'id': 2,
                    'width': 150,
                    'height': 150,
                    'file_name': 'image2.jpg',
                    'license': 1,
                    'date_captured': '2024-01-01 11:00:00',
                },
            ],
            'annotations': [
                {'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [10, 20, 30, 40], 'area': 1200, 'iscrowd': 0},
                {'id': 2, 'image_id': 1, 'category_id': 2, 'bbox': [50, 60, 20, 30], 'area': 600, 'iscrowd': 0},
                {'id': 3, 'image_id': 2, 'category_id': 1, 'bbox': [5, 5, 40, 40], 'area': 1600, 'iscrowd': 0},
            ],
            'categories': [
                {'id': 1, 'name': 'person', 'supercategory': 'human'},
                {'id': 2, 'name': 'car', 'supercategory': 'vehicle'},
            ],
        }

    def create_test_images(self, images_dir: Path) -> None:
        """Create test images in the specified directory."""
        images_dir.mkdir(exist_ok=True)

        # Create image1.jpg (100x100)
        img1 = PIL.Image.new('RGB', (100, 100), color='red')
        img1.save(images_dir / 'image1.jpg')

        # Create image2.jpg (150x150)
        img2 = PIL.Image.new('RGB', (150, 150), color='blue')
        img2.save(images_dir / 'image2.jpg')

    def test_import_coco_basic(self, reset_db: None) -> None:
        """Test basic COCO import functionality."""
        pxt.create_dir('test')
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            images_dir = tmp_path / 'images'
            annotations_file = tmp_path / 'annotations.json'

            # Create test data
            coco_data = self.create_sample_coco_data()
            self.create_test_images(images_dir)

            # Write COCO JSON
            with open(annotations_file, 'w', encoding='utf-8') as f:
                json.dump(coco_data, f)

            # Import COCO data
            tbl = pxt.io.import_coco('test.coco_import', annotations_file, images_dir)

            # Verify table structure
            assert tbl.count() == 2  # Two images

            # Check schema
            schema = tbl.schema
            assert 'image' in schema
            assert 'annotations' in schema
            assert 'image_id' in schema
            assert 'file_name' in schema
            assert 'width' in schema
            assert 'height' in schema

            # Verify data
            rows = list(tbl.select())
            assert len(rows) == 2

            # Check first image
            row1 = next(r for r in rows if r['image_id'] == 1)
            assert row1['file_name'] == 'image1.jpg'
            assert row1['width'] == 100
            assert row1['height'] == 100
            assert isinstance(row1['image'], PIL.Image.Image)
            assert len(row1['annotations']) == 2  # Two annotations for image1

            # Check annotations format
            ann1 = row1['annotations'][0]
            assert 'bbox' in ann1
            assert 'category' in ann1
            assert ann1['category'] in ['person', 'car']
            assert len(ann1['bbox']) == 4

            # Check second image
            row2 = next(r for r in rows if r['image_id'] == 2)
            assert row2['file_name'] == 'image2.jpg'
            assert row2['width'] == 150
            assert row2['height'] == 150
            assert len(row2['annotations']) == 1  # One annotation for image2

    def test_import_coco_with_schema_overrides(self, reset_db: None) -> None:
        """Test COCO import with schema overrides."""
        pxt.create_dir('test')
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            images_dir = tmp_path / 'images'
            annotations_file = tmp_path / 'annotations.json'

            # Create test data
            coco_data = self.create_sample_coco_data()
            self.create_test_images(images_dir)

            # Write COCO JSON
            with open(annotations_file, 'w', encoding='utf-8') as f:
                json.dump(coco_data, f)

            # Import with schema overrides
            tbl = pxt.io.import_coco(
                'test.coco_schema_override', annotations_file, images_dir, schema_overrides={'image_id': pxt.String}
            )

            # Verify schema override was applied
            schema = tbl.schema
            assert schema['image_id'].is_string_type()

    def test_import_coco_error_missing_json(self, reset_db: None) -> None:
        """Test error handling when COCO JSON file is missing."""
        pxt.create_dir('test')
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            missing_file = tmp_path / 'missing.json'
            images_dir = tmp_path / 'images'
            images_dir.mkdir()

            with pytest.raises(excs.Error, match='COCO JSON file not found'):
                pxt.io.import_coco('test.missing_json', missing_file, images_dir)

    def test_import_coco_error_missing_images_dir(self, reset_db: None) -> None:
        """Test error handling when images directory is missing."""
        pxt.create_dir('test')
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            annotations_file = tmp_path / 'annotations.json'
            missing_dir = tmp_path / 'missing_images'

            # Create minimal COCO JSON
            with open(annotations_file, 'w', encoding='utf-8') as f:
                json.dump(self.create_sample_coco_data(), f)

            with pytest.raises(excs.Error, match='Images directory not found'):
                pxt.io.import_coco('test.missing_dir', annotations_file, missing_dir)

    def test_import_coco_error_invalid_json(self, reset_db: None) -> None:
        """Test error handling with invalid JSON."""
        pxt.create_dir('test')
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            annotations_file = tmp_path / 'invalid.json'
            images_dir = tmp_path / 'images'
            images_dir.mkdir()

            # Write invalid JSON
            with open(annotations_file, 'w', encoding='utf-8') as f:
                f.write('{ invalid json }')

            with pytest.raises(excs.Error, match='Invalid JSON in COCO file'):
                pxt.io.import_coco('test.invalid_json', annotations_file, images_dir)

    def test_import_coco_error_missing_required_fields(self, reset_db: None) -> None:
        """Test error handling when required COCO fields are missing."""
        pxt.create_dir('test')
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            annotations_file = tmp_path / 'incomplete.json'
            images_dir = tmp_path / 'images'
            images_dir.mkdir()

            # Create incomplete COCO data (missing 'categories')
            incomplete_data: dict[str, Any] = {
                'images': [],
                'annotations': [],
                # Missing 'categories'
            }

            with open(annotations_file, 'w', encoding='utf-8') as f:
                json.dump(incomplete_data, f)

            with pytest.raises(excs.Error, match='Missing required field "categories"'):
                pxt.io.import_coco('test.incomplete', annotations_file, images_dir)

    def test_import_coco_error_missing_image_file(self, reset_db: None) -> None:
        """Test error handling when an image file is missing."""
        pxt.create_dir('test')
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            annotations_file = tmp_path / 'annotations.json'
            images_dir = tmp_path / 'images'
            images_dir.mkdir()

            # Create COCO data but don't create the image files
            coco_data = self.create_sample_coco_data()
            with open(annotations_file, 'w', encoding='utf-8') as f:
                json.dump(coco_data, f)

            with pytest.raises(excs.Error, match='Image file not found'):
                pxt.io.import_coco('test.missing_image', annotations_file, images_dir)

    def test_import_coco_empty_annotations(self, reset_db: None) -> None:
        """Test COCO import with images that have no annotations."""
        pxt.create_dir('test')
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            images_dir = tmp_path / 'images'
            annotations_file = tmp_path / 'annotations.json'

            # Create COCO data with no annotations
            coco_data = {
                'images': [{'id': 1, 'width': 100, 'height': 100, 'file_name': 'image1.jpg'}],
                'annotations': [],  # No annotations
                'categories': [{'id': 1, 'name': 'person'}],
            }

            self.create_test_images(images_dir)

            with open(annotations_file, 'w', encoding='utf-8') as f:
                json.dump(coco_data, f)

            # Import should succeed
            tbl = pxt.io.import_coco('test.empty_annotations', annotations_file, images_dir)

            # Verify data
            rows = list(tbl.select())
            assert len(rows) == 1
            assert len(rows[0]['annotations']) == 0  # Empty annotations list

    def test_import_coco_integration_with_export(self, reset_db: None) -> None:
        """Test that imported COCO data can be exported back to COCO format."""
        pxt.create_dir('test')
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            images_dir = tmp_path / 'images'
            annotations_file = tmp_path / 'annotations.json'

            # Create test data
            coco_data = self.create_sample_coco_data()
            self.create_test_images(images_dir)

            with open(annotations_file, 'w', encoding='utf-8') as f:
                json.dump(coco_data, f)

            # Import COCO data
            tbl = pxt.io.import_coco('test.coco_roundtrip', annotations_file, images_dir)

            # Create a query in the format expected by to_coco_dataset
            query = tbl.select({'image': tbl.image, 'annotations': tbl.annotations})

            # Export to COCO format
            export_path = query.to_coco_dataset()

            # Verify the export succeeded
            assert export_path.exists()
            assert export_path.is_file()

            # Load and verify the exported data
            with open(export_path, 'r', encoding='utf-8') as f:
                exported_data = json.load(f)

            assert 'images' in exported_data
            assert 'annotations' in exported_data
            assert 'categories' in exported_data
            assert len(exported_data['images']) == 2  # Two images

    def test_import_coco_with_unknown_categories(self, reset_db: None) -> None:
        """Test COCO import with annotations referencing unknown category IDs."""
        pxt.create_dir('test')
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            images_dir = tmp_path / 'images'
            annotations_file = tmp_path / 'annotations.json'

            # Create COCO data with annotation referencing unknown category
            coco_data = {
                'images': [{'id': 1, 'width': 100, 'height': 100, 'file_name': 'image1.jpg'}],
                'annotations': [
                    {
                        'id': 1,
                        'image_id': 1,
                        'category_id': 999,  # Unknown category ID
                        'bbox': [10, 20, 30, 40],
                    }
                ],
                'categories': [{'id': 1, 'name': 'person'}],
            }

            self.create_test_images(images_dir)

            with open(annotations_file, 'w', encoding='utf-8') as f:
                json.dump(coco_data, f)

            # Import should succeed but use fallback category name
            tbl = pxt.io.import_coco('test.unknown_category', annotations_file, images_dir)

            # Verify unknown category is handled
            rows = list(tbl.select())
            assert len(rows) == 1
            assert len(rows[0]['annotations']) == 1
            assert rows[0]['annotations'][0]['category'] == 'unknown_999'

    def test_import_coco_different_image_modes(self, reset_db: None) -> None:
        """Test COCO import with different image color modes (RGBA, grayscale, etc.)."""
        pxt.create_dir('test')
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            images_dir = tmp_path / 'images'
            annotations_file = tmp_path / 'annotations.json'
            images_dir.mkdir()

            # Create images with different modes
            # RGBA image
            rgba_img = PIL.Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))
            rgba_img.save(images_dir / 'rgba_image.png')

            # Grayscale image
            gray_img = PIL.Image.new('L', (100, 100), color=128)
            gray_img.save(images_dir / 'gray_image.jpg')

            # Create COCO data
            coco_data = {
                'images': [
                    {'id': 1, 'width': 100, 'height': 100, 'file_name': 'rgba_image.png'},
                    {'id': 2, 'width': 100, 'height': 100, 'file_name': 'gray_image.jpg'},
                ],
                'annotations': [],
                'categories': [],
            }

            with open(annotations_file, 'w', encoding='utf-8') as f:
                json.dump(coco_data, f)

            # Import should succeed and convert all images to RGB
            tbl = pxt.io.import_coco('test.image_modes', annotations_file, images_dir)

            # Verify all images are converted to RGB
            rows = list(tbl.select())
            assert len(rows) == 2
            for row in rows:
                assert row['image'].mode == 'RGB'
