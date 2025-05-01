import time

import pixeltable as pxt
from tabulate import tabulate  # type: ignore


def run_user_workflow_test() -> None:
    """Performs a Pixeltable workflow test with timing:
    Imports, counts, embeds, queries, adds col, updates, deletes, views.
    """
    total_start_time = time.monotonic()
    # Initialize duration variables
    duration_step0 = 0.0
    duration_step1 = 0.0
    duration_step1b = 0.0
    duration_step2 = 0.0
    duration_step3 = 0.0
    duration_step3b = 0.0
    duration_step4 = 0.0
    duration_step5 = 0.0
    duration_step6 = 0.0
    duration_step7 = 0.0
    duration_step8 = 0.0

    # --- Step 0: Setup Pixeltable Directory ---
    print('\n--- Step 0: Setup Pixeltable Directory ---')
    step0_start_time = time.monotonic()
    try:
        pxt.drop_dir('user_workflow_hf_import_test_dir', force=True)
        pxt.create_dir('user_workflow_hf_import_test_dir')
    except Exception as e:
        print(f"   ERROR setting up directory 'user_workflow_hf_import_test_dir': {e}")
        return
    finally:
        duration_step0 = time.monotonic() - step0_start_time
        print(f'--- Step 0 Duration: {duration_step0:.2f} seconds ---')

    try:
        # --- Step 1: Import Hugging Face Dataset (All Splits) ---
        print('\n--- Step 1: Import Hugging Face Dataset (All Splits) ---')
        import datasets  # type: ignore

        images_tbl = None
        step1_start_time = time.monotonic()
        dataset_dict = None
        total_rows_in_source = 0
        try:
            print("   Loading all splits for dataset 'lscpku/RefCOCO_rec'...")
            dataset_dict = datasets.load_dataset('lscpku/RefCOCO_rec')
            print(f'   DatasetDict loaded. Splits: {list(dataset_dict.keys())}')
            total_rows_in_source = sum(len(split) for split in dataset_dict.values())
            print(f'   Total rows across all splits in source: {total_rows_in_source}')

            images_tbl = pxt.io.import_huggingface_dataset(
                'user_workflow_hf_import_test_dir.hf_images',
                dataset_dict,
                primary_key='question_id',
                on_error='ignore',
                comment='Test table for user workflow (all splits)',
                media_validation='on_read',
                column_name_for_split='split_name',
                if_exists='replace',
            )
            print('   Import initiated...')

        except Exception as e:
            print(f'   ERROR importing Hugging Face dataset: {e}')
            print("   Ensure dataset 'lscpku/RefCOCO_rec' exists and dependencies are installed.")
            images_tbl = None
        finally:
            duration_step1 = time.monotonic() - step1_start_time
            print(f'--- Step 1 Duration: {duration_step1:.2f} seconds ---')
            if images_tbl is None:
                print('   Halting test due to failure in Step 1.')

        # --- Step 1b: Count Imported Rows ---
        print('\n--- Step 1b: Count Imported Rows ---')
        step1b_start_time = time.monotonic()
        imported_rows_count = 0
        count_successful = False
        try:
            imported_rows_count = images_tbl.count()
            print(f'   Final imported row count: {imported_rows_count}')
            if imported_rows_count < total_rows_in_source:
                print(
                    f'   Warning: Final count ({imported_rows_count}) is less than source ({total_rows_in_source}) '
                    f"due to on_error='ignore' during import."
                )
            count_successful = True
        except Exception as e:
            print(f'   ERROR counting rows after import: {e}')
        finally:
            duration_step1b = time.monotonic() - step1b_start_time
            print(f'--- Step 1b Duration: {duration_step1b:.2f} seconds ---')
            if not count_successful:
                print('   Halting test due to failure in Step 1b.')

        if not count_successful:
            print('   Halting test due to failure in Step 1b.')
            return

        # --- Step 2: Create Embedding Index ---
        print('\n--- Step 2: Create Embedding Index ---')
        from pixeltable.functions.huggingface import clip

        step2_start_time = time.monotonic()
        index_creation_successful = False
        try:
            images_tbl.add_embedding_index('image', embedding=clip.using(model_id='openai/clip-vit-base-patch32'))
            _ = images_tbl.count()
            index_creation_successful = True
        except Exception as e:
            print(f'   ERROR creating embedding index: {e}')
        finally:
            duration_step2 = time.monotonic() - step2_start_time
            print(f'--- Step 2 Duration: {duration_step2:.2f} seconds ---')
            if not index_creation_successful:
                print('   Halting test due to failure in Step 2.')

        if not index_creation_successful:
            print('   Halting test due to failure in Step 2.')
            return

        # --- Step 3: Similarity Search Query ---
        print('\n--- Step 3: Similarity Search Query ---')
        step3_start_time = time.monotonic()
        try:
            query_text = 'a bowl of broccoli'
            sim_expr = images_tbl.image.similarity(query_text)
            select_kwargs = {'similarity': sim_expr}
            results_rs = (
                images_tbl.order_by(sim_expr, asc=False)
                .limit(5)
                .select(images_tbl.image, images_tbl.answer, **select_kwargs)
                .collect()
            )
            print(f'\n--- Similarity Query Results (Top {min(len(results_rs), 5)}) ---')
            print(results_rs)
        except Exception as e:
            print(f'   ERROR during similarity query execution: {e}')
        finally:
            duration_step3 = time.monotonic() - step3_start_time
            print(f'--- Step 3 Duration: {duration_step3:.2f} seconds ---')

        # --- Step 3b: Metadata Filter Query ---
        print('\n--- Step 3b: Metadata Filter Query ---')
        step3b_start_time = time.monotonic()
        filter_results_rs = None
        try:
            filter_width = 600
            filter_results_rs = (
                images_tbl.where(images_tbl.image_width > filter_width)
                .select(images_tbl.file_name, images_tbl.image_width)
                .limit(5)
                .collect()
            )
            print(f'\n--- Metadata Filter Query Results (width > {filter_width}, {len(filter_results_rs)} rows) ---')
            print(filter_results_rs)
        except Exception as e:
            print(f'   ERROR during filter query execution: {e}')
        finally:
            duration_step3b = time.monotonic() - step3b_start_time
            print(f'--- Step 3b Duration: {duration_step3b:.2f} seconds ---')

        # --- Step 4: Add Computed Column ---
        print('\n--- Step 4: Add Computed Column (is_jpg) ---')
        step4_start_time = time.monotonic()
        computed_col_successful = False
        try:
            images_tbl.add_computed_column(is_jpg=images_tbl.file_name.endswith('.jpg'))
            print("   Added computed column 'is_jpg'. Waiting for population...")
            print(images_tbl.head(3))
            computed_col_successful = True
        except Exception as e:
            print(f'   ERROR adding computed column: {e}')
        finally:
            duration_step4 = time.monotonic() - step4_start_time
            print(f'--- Step 4 Duration: {duration_step4:.2f} seconds ---')
            if not computed_col_successful:
                print('   Halting test due to failure in Step 4.')

        if not computed_col_successful:
            print('   Halting test due to failure in Step 4.')
            return

        # --- Step 5: Update Rows ---
        print('\n--- Step 5: Update Rows ---')
        step5_start_time = time.monotonic()
        updated_rows_count = 0
        try:
            update_value = 'Updated Answer: Wide Image'
            update_filter = images_tbl.image_width > 520
            print(f"   Updating 'answer' to '{update_value}' where image_width > 520...")
            status = images_tbl.update({'answer': update_value}, where=update_filter, cascade=False)
            updated_rows_count = status.num_rows
            print(f'   Update operation completed. Rows updated: {updated_rows_count}')
        except Exception as e:
            print(f'   ERROR updating rows: {e}')
        finally:
            duration_step5 = time.monotonic() - step5_start_time
            print(f'--- Step 5 Duration: {duration_step5:.2f} seconds ---')

        # --- Step 6: Delete Rows ---
        print('\n--- Step 6: Delete Rows ---')
        step6_start_time = time.monotonic()
        deleted_rows_count = 0
        try:
            delete_filter = images_tbl.image_width >= 480
            print('   Deleting rows where image_width >= 480...')
            status = images_tbl.delete(where=delete_filter)
            deleted_rows_count = status.num_rows
            print(f'   Delete operation completed. Rows deleted: {deleted_rows_count}')
        except Exception as e:
            print(f'   ERROR deleting rows: {e}')
        finally:
            duration_step6 = time.monotonic() - step6_start_time
            print(f'--- Step 6 Duration: {duration_step6:.2f} seconds ---')

        # --- Step 7: Create View (Filtered, No Image) ---
        print('\n--- Step 7: Create View (Filtered, No Image) ---')
        view_name = 'user_workflow_hf_import_test_dir.filtered_no_image_view'
        filtered_no_image_view = None
        step7_start_time = time.monotonic()
        try:
            view_filter = images_tbl.image_width < 480
            filtered_table_for_view = images_tbl.where(view_filter).select(
                images_tbl.file_name,
                images_tbl.answer,
                images_tbl.image_width,
                images_tbl.split_name,
                images_tbl.is_jpg,
            )
            print(
                f"   Creating view '{view_name}' from table filtered where image_width < 480 (excluding image col)..."
            )
            filtered_no_image_view = pxt.create_view(view_name, filtered_table_for_view, if_exists='replace')
            print(f"   View '{view_name}' created successfully.")
        except Exception as e:
            print(f'   ERROR creating view: {e}')
            filtered_no_image_view = None
        finally:
            duration_step7 = time.monotonic() - step7_start_time
            print(f'--- Step 7 Duration: {duration_step7:.2f} seconds ---')
            if filtered_no_image_view is None:
                print('   Halting test due to failure in Step 7.')

        if filtered_no_image_view is None:
            print('   Halting test due to failure in Step 7.')
            return

        # --- Step 8: Query View (Filtered, No Image) ---
        print('\n--- Step 8: Query View (Filtered, No Image) ---')
        step8_start_time = time.monotonic()
        try:
            view_count = filtered_no_image_view.count()
            print(f"   Querying view '{view_name}'. Count: {view_count}")
            print('   View head:')
            print(filtered_no_image_view.head(3))
        except Exception as e:
            print(f'   ERROR querying view: {e}')
        finally:
            duration_step8 = time.monotonic() - step8_start_time
            print(f'--- Step 8 Duration: {duration_step8:.2f} seconds ---')

        # --- Final Summary ---
        total_duration = time.monotonic() - total_start_time
        print("\n--- Test finished. Pixeltable resources are in directory: 'user_workflow_hf_import_test_dir' ---")

        # Print Key Test Parameters
        print('\n--- Test Parameters ---')
        print(f'Dataset:                  {"lscpku/RefCOCO_rec"}')
        print('Rows Imported:            All Splits')
        print(f'Embedding Model:        {"openai/clip-vit-base-patch32"}')

        # Print summary table
        print('\n--- Execution Time Summary ---')
        headers = ['Step', 'Duration (min)', 'Duration (s)', 'Duration (ms)', 'Rows Affected']
        data = [
            [
                '0: Setup Directory',
                f'{duration_step0 / 60:.2f}',
                f'{duration_step0:.2f}',
                f'{duration_step0 * 1000:.0f}',
                '-',
            ],
            [
                '1: Import Dataset',
                f'{duration_step1 / 60:.2f}',
                f'{duration_step1:.2f}',
                f'{duration_step1 * 1000:.0f}',
                '-',
            ],
            [
                '1b: Count Imported Rows',
                f'{duration_step1b / 60:.2f}',
                f'{duration_step1b:.2f}',
                f'{duration_step1b * 1000:.0f}',
                str(imported_rows_count) if count_successful else 'FAIL',
            ],
            [
                '2: Create Embedding Index',
                f'{duration_step2 / 60:.2f}',
                f'{duration_step2:.2f}',
                f'{duration_step2 * 1000:.0f}',
                '-',
            ],
            [
                '3: Similarity Search',
                f'{duration_step3 / 60:.2f}',
                f'{duration_step3:.2f}',
                f'{duration_step3 * 1000:.0f}',
                '-',
            ],
            [
                '3b: Metadata Filter',
                f'{duration_step3b / 60:.2f}',
                f'{duration_step3b:.2f}',
                f'{duration_step3b * 1000:.0f}',
                '-',
            ],
            [
                '4: Add Computed Column',
                f'{duration_step4 / 60:.2f}',
                f'{duration_step4:.2f}',
                f'{duration_step4 * 1000:.0f}',
                '-',
            ],
            [
                '5: Update Rows',
                f'{duration_step5 / 60:.2f}',
                f'{duration_step5:.2f}',
                f'{duration_step5 * 1000:.0f}',
                updated_rows_count,
            ],
            [
                '6: Delete Rows',
                f'{duration_step6 / 60:.2f}',
                f'{duration_step6:.2f}',
                f'{duration_step6 * 1000:.0f}',
                deleted_rows_count,
            ],
            [
                '7: Create View (No Img)',
                f'{duration_step7 / 60:.2f}',
                f'{duration_step7:.2f}',
                f'{duration_step7 * 1000:.0f}',
                str(filtered_no_image_view.count()) if filtered_no_image_view is not None else 'FAIL',
            ],
            [
                '8: Query View (No Img)',
                f'{duration_step8 / 60:.2f}',
                f'{duration_step8:.2f}',
                f'{duration_step8 * 1000:.0f}',
                '-',
            ],
        ]
        total_row = [
            'Total Execution Time',
            f'{total_duration / 60:.2f}',
            f'{total_duration:.2f}',
            f'{total_duration * 1000:.0f}',
            '-',
        ]

        print(tabulate([*data, total_row], headers=headers, tablefmt='grid'))

        print('\nUser workflow test completed.')

    finally:
        # --- Cleanup ---
        print('\n--- Final Cleanup: Dropping Test Directory ---')
        try:
            pxt.drop_dir('user_workflow_hf_import_test_dir', force=True)
            print("   Directory 'user_workflow_hf_import_test_dir' dropped successfully.")
        except Exception as e:
            print(f"   ERROR dropping directory 'user_workflow_hf_import_test_dir': {e}")


if __name__ == '__main__':
    run_user_workflow_test()
