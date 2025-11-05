// Generated TypeScript SDK for Pixeltable
// This file is auto-generated from the Pixeltable public API

// This is a placeholder implementation.
// You will need to implement the actual API client logic
// (e.g., HTTP requests to a Pixeltable server)

class PixeltableClient {
  private baseUrl: string;
  private apiKey?: string;

  constructor(baseUrl: string, apiKey?: string) {
    this.baseUrl = baseUrl;
    this.apiKey = apiKey;
  }

  private async request<T>(endpoint: string, method: string, body?: any): Promise<T> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };

    if (this.apiKey) {
      headers["Authorization"] = `Bearer ${this.apiKey}`;
    }

    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      method,
      headers,
      body: body ? JSON.stringify(body) : undefined,
    });

    if (!response.ok) {
      throw new Error(`Pixeltable API error: ${response.statusText}`);
    }

    return response.json();
  }

  async get_public_api_registry(): Promise<any> {
    // TODO: Implement get_public_api_registry
    return this.request(`/api/get_public_api_registry`, "POST", {  });
  }

  async is_public_api(obj: any): Promise<any> {
    // TODO: Implement is_public_api
    return this.request(`/api/is_public_api`, "POST", { obj });
  }

  async array(elements: any): Promise<any> {
    // TODO: Implement array
    return this.request(`/api/array`, "POST", { elements });
  }

  async configure_logging(to_stdout: any, level: any, add: any, remove: any): Promise<any> {
    // TODO: Implement configure_logging
    return this.request(`/api/configure_logging`, "POST", { to_stdout, level, add, remove });
  }

  async create_dir(path: any, if_exists: any, parents: any): Promise<any> {
    // TODO: Implement create_dir
    return this.request(`/api/create_dir`, "POST", { path, if_exists, parents });
  }

  async create_snapshot(path_str: any, base: any, additional_columns: any, iterator: any, num_retained_versions: any, comment: any, media_validation: any, if_exists: any): Promise<any> {
    // TODO: Implement create_snapshot
    return this.request(`/api/create_snapshot`, "POST", { path_str, base, additional_columns, iterator, num_retained_versions, comment, media_validation, if_exists });
  }

  async create_table(): Promise<any> {
    // TODO: Implement create_table
    return this.request(`/api/create_table`, "POST", {  });
  }

  async create_view(path: any, base: any, additional_columns: any, is_snapshot: any, iterator: any, num_retained_versions: any, comment: any, media_validation: any, if_exists: any): Promise<any> {
    // TODO: Implement create_view
    return this.request(`/api/create_view`, "POST", { path, base, additional_columns, is_snapshot, iterator, num_retained_versions, comment, media_validation, if_exists });
  }

  async drop_dir(path: any, force: any, if_not_exists: any): Promise<any> {
    // TODO: Implement drop_dir
    return this.request(`/api/drop_dir`, "POST", { path, force, if_not_exists });
  }

  async drop_table(table: any, force: any, if_not_exists: any): Promise<any> {
    // TODO: Implement drop_table
    return this.request(`/api/drop_table`, "POST", { table, force, if_not_exists });
  }

  async get_dir_contents(): Promise<any> {
    // TODO: Implement get_dir_contents
    return this.request(`/api/get_dir_contents`, "POST", {  });
  }

  async get_table(path: any, if_not_exists: any): Promise<any> {
    // TODO: Implement get_table
    return this.request(`/api/get_table`, "POST", { path, if_not_exists });
  }

  async init(config_overrides: any): Promise<any> {
    // TODO: Implement init
    return this.request(`/api/init`, "POST", { config_overrides });
  }

  async list_dirs(path: any, recursive: any): Promise<any> {
    // TODO: Implement list_dirs
    return this.request(`/api/list_dirs`, "POST", { path, recursive });
  }

  async list_functions(): Promise<any> {
    // TODO: Implement list_functions
    return this.request(`/api/list_functions`, "POST", {  });
  }

  async list_tables(dir_path: any, recursive: any): Promise<any> {
    // TODO: Implement list_tables
    return this.request(`/api/list_tables`, "POST", { dir_path, recursive });
  }

  async ls(path: any): Promise<any> {
    // TODO: Implement ls
    return this.request(`/api/ls`, "POST", { path });
  }

  async move(path: any, new_path: any, if_exists: any, if_not_exists: any): Promise<any> {
    // TODO: Implement move
    return this.request(`/api/move`, "POST", { path, new_path, if_exists, if_not_exists });
  }

  async publish(source: any, destination_uri: any, bucket_name: any, access: any): Promise<any> {
    // TODO: Implement publish
    return this.request(`/api/publish`, "POST", { source, destination_uri, bucket_name, access });
  }

  async replicate(remote_uri: any, local_path: any): Promise<any> {
    // TODO: Implement replicate
    return this.request(`/api/replicate`, "POST", { remote_uri, local_path });
  }

  async tool(fn: any, name: any, description: any): Promise<any> {
    // TODO: Implement tool
    return this.request(`/api/tool`, "POST", { fn, name, description });
  }

  async tools(args: any): Promise<any> {
    // TODO: Implement tools
    return this.request(`/api/tools`, "POST", { args });
  }

  async import_json(tbl_path: any, filepath_or_url: any, schema_overrides: any, primary_key: any, num_retained_versions: any, comment: any, kwargs: any): Promise<any> {
    // TODO: Implement import_json
    return this.request(`/api/import_json`, "POST", { tbl_path, filepath_or_url, schema_overrides, primary_key, num_retained_versions, comment, kwargs });
  }

  async import_rows(tbl_path: any, rows: any, schema_overrides: any, primary_key: any, num_retained_versions: any, comment: any): Promise<any> {
    // TODO: Implement import_rows
    return this.request(`/api/import_rows`, "POST", { tbl_path, rows, schema_overrides, primary_key, num_retained_versions, comment });
  }

  async create_label_studio_project(t: any, label_config: any, name: any, title: any, media_import_method: any, col_mapping: any, sync_immediately: any, s3_configuration: any, kwargs: any): Promise<any> {
    // TODO: Implement create_label_studio_project
    return this.request(`/api/create_label_studio_project`, "POST", { t, label_config, name, title, media_import_method, col_mapping, sync_immediately, s3_configuration, kwargs });
  }

  async export_images_as_fo_dataset(): Promise<any> {
    // TODO: Implement export_images_as_fo_dataset
    return this.request(`/api/export_images_as_fo_dataset`, "POST", {  });
  }

  async import_huggingface_dataset(): Promise<any> {
    // TODO: Implement import_huggingface_dataset
    return this.request(`/api/import_huggingface_dataset`, "POST", {  });
  }

  async import_csv(tbl_name: any, filepath_or_buffer: any, schema_overrides: any, primary_key: any, num_retained_versions: any, comment: any, kwargs: any): Promise<any> {
    // TODO: Implement import_csv
    return this.request(`/api/import_csv`, "POST", { tbl_name, filepath_or_buffer, schema_overrides, primary_key, num_retained_versions, comment, kwargs });
  }

  async import_excel(tbl_name: any, io: any, schema_overrides: any, primary_key: any, num_retained_versions: any, comment: any, kwargs: any): Promise<any> {
    // TODO: Implement import_excel
    return this.request(`/api/import_excel`, "POST", { tbl_name, io, schema_overrides, primary_key, num_retained_versions, comment, kwargs });
  }

  async import_pandas(tbl_name: any, df: any, schema_overrides: any, primary_key: any, num_retained_versions: any, comment: any): Promise<any> {
    // TODO: Implement import_pandas
    return this.request(`/api/import_pandas`, "POST", { tbl_name, df, schema_overrides, primary_key, num_retained_versions, comment });
  }

  async export_parquet(table_or_df: any, parquet_path: any, partition_size_bytes: any, inline_images: any): Promise<any> {
    // TODO: Implement export_parquet
    return this.request(`/api/export_parquet`, "POST", { table_or_df, parquet_path, partition_size_bytes, inline_images });
  }

  async import_parquet(table: any, parquet_path: any, schema_overrides: any, primary_key: any, kwargs: any): Promise<any> {
    // TODO: Implement import_parquet
    return this.request(`/api/import_parquet`, "POST", { table, parquet_path, schema_overrides, primary_key, kwargs });
  }

  async _overload_dummy(args: any, kwds: any): Promise<any> {
    // TODO: Implement _overload_dummy
    return this.request(`/api/_overload_dummy`, "POST", { args, kwds });
  }

}

export default PixeltableClient;