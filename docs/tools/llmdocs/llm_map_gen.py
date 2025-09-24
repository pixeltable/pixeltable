"""LLM Map generator - creates JSON-LD structured data for API documentation."""

import json
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import importlib
from docstring_parser import parse as parse_docstring

try:
    from pyld import jsonld
    HAS_PYLD = True
except ImportError:
    HAS_PYLD = False
    print("⚠️ pyld not installed - JSON-LD flattening disabled")
    print("   Install with: pip install PyLD")


class LLMMapGenerator:
    """Generate JSON-LD structured data map for LLM consumption."""
    
    def __init__(self, output_dir: Path, version: str = "main"):
        """Initialize with output directory and version."""
        self.output_dir = output_dir
        self.version = version
        self.base_url = f"https://docs.pixeltable.com/sdk/{version}"
        self.api_map = {
            "@context": "https://schema.org",
            "@type": "SoftwareSourceCode",
            "name": "Pixeltable SDK",
            "version": version,
            "url": self.base_url,
            "programmingLanguage": "Python",
            "dateModified": datetime.now().isoformat(),
            "hasPart": []
        }
    
    def add_module(self, module_path: str, children: Optional[List[str]] = None) -> Dict:
        """Add a module to the LLM map."""
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            error_entry = self._create_error_entry(module_path, str(e), "Module")
            self.api_map["hasPart"].append(error_entry)
            return error_entry
        
        doc = inspect.getdoc(module) or ""
        parsed = parse_docstring(doc) if doc else None
        
        module_entry = {
            "@type": "SoftwareSourceCode",
            "@id": f"pxt:{module_path}",  # Use module path as stable identifier
            "name": module_path,
            "description": parsed.short_description if parsed else "",
            "programmingLanguage": "Python",
            "url": f"{self.base_url}/{module_path.replace('.', '/')}",  # Documentation URL
            "codeRepository": f"https://github.com/pixeltable/pixeltable/tree/{self.version}",
            "hasPart": []
        }
        
        # Add module contents
        if children:
            # Only document specified children
            for child_name in children:
                if hasattr(module, child_name):
                    child = getattr(module, child_name)
                    child_entry = self._document_item(child, child_name, module_path)
                    if child_entry:
                        module_entry["hasPart"].append(child_entry)
        else:
            # Document all public items
            for name in dir(module):
                if not name.startswith('_'):
                    try:
                        obj = getattr(module, name)
                        item_entry = self._document_item(obj, name, module_path)
                        if item_entry:
                            module_entry["hasPart"].append(item_entry)
                    except AttributeError:
                        continue
        
        # Add to main API map
        self.api_map["hasPart"].append(module_entry)
        return module_entry
    
    def add_class(self, class_path: str, children: Optional[List[str]] = None) -> Dict:
        """Add a class to the LLM map."""
        parts = class_path.split('.')
        module_path = '.'.join(parts[:-1])
        class_name = parts[-1]
        
        try:
            module = importlib.import_module(module_path)
            if not hasattr(module, class_name):
                error_entry = self._create_error_entry(class_path, f"Class {class_name} not found", "Class")
                self.api_map["hasPart"].append(error_entry)
                return error_entry
            cls = getattr(module, class_name)
            if not inspect.isclass(cls):
                error_entry = self._create_error_entry(class_path, f"{class_name} is not a class", "Class")
                self.api_map["hasPart"].append(error_entry)
                return error_entry
        except ImportError as e:
            error_entry = self._create_error_entry(class_path, str(e), "Class")
            self.api_map["hasPart"].append(error_entry)
            return error_entry
        
        doc = inspect.getdoc(cls) or ""
        parsed = parse_docstring(doc) if doc else None
        
        class_entry = {
            "@type": "Class",
            "@id": f"pxt:{class_path}",  # Use class path as stable identifier
            "name": class_name,
            "description": parsed.short_description if parsed else "",
            "url": f"{self.base_url}/{class_path.replace('.', '/')}",  # Documentation URL
            "memberOf": {
                "@type": "SoftwareSourceCode",
                "@id": f"pxt:{module_path}"
            },
            "hasPart": []
        }
        
        # Add source location
        source_info = self._get_source_location(cls)
        if source_info:
            class_entry["sourceLocation"] = source_info
        
        # Add constructor
        try:
            sig = inspect.signature(cls.__init__)
            class_entry["constructor"] = self._format_signature_json(sig)
        except:
            pass
        
        # Add methods
        if children:
            # Only specified methods
            for method_name in children:
                if hasattr(cls, method_name):
                    method = getattr(cls, method_name)
                    method_entry = self._document_method(method, method_name, class_name)
                    if method_entry:
                        class_entry["hasPart"].append(method_entry)
        else:
            # All public methods
            for name, obj in inspect.getmembers(cls):
                if not name.startswith('_') or name in ['__init__', '__call__']:
                    if inspect.ismethod(obj) or inspect.isfunction(obj):
                        if name != '__init__':  # Already documented as constructor
                            method_entry = self._document_method(obj, name, class_name)
                            if method_entry:
                                class_entry["hasPart"].append(method_entry)
        
        # Add to main API map
        self.api_map["hasPart"].append(class_entry)
        return class_entry
    
    def add_type(self, type_path: str) -> Dict:
        """Add a type to the LLM map."""
        # Types are usually just type hints, not actual API elements
        # Return a minimal entry
        type_entry = {
            "@type": "DataType",
            "@id": f"pxt:{type_path}",
            "name": type_path.split('.')[-1],
            "description": "Type definition"
        }
        self.api_map["hasPart"].append(type_entry)
        return type_entry
    
    def add_function(self, function_path: str) -> Dict:
        """Add a function to the LLM map."""
        parts = function_path.split('.')
        module_path = '.'.join(parts[:-1])
        func_name = parts[-1]
        
        try:
            module = importlib.import_module(module_path)
            if not hasattr(module, func_name):
                error_entry = self._create_error_entry(function_path, f"Function {func_name} not found", "Function")
                self.api_map["hasPart"].append(error_entry)
                return error_entry
            func = getattr(module, func_name)
            if not callable(func):
                error_entry = self._create_error_entry(function_path, f"{func_name} is not callable", "Function")
                self.api_map["hasPart"].append(error_entry)
                return error_entry
        except ImportError as e:
            error_entry = self._create_error_entry(function_path, str(e), "Function")
            self.api_map["hasPart"].append(error_entry)
            return error_entry
        
        func_entry = self._document_function(func, func_name, module_path)
        self.api_map["hasPart"].append(func_entry)
        return func_entry
    
    def _document_item(self, obj: Any, name: str, module_path: str) -> Optional[Dict]:
        """Document a single item."""
        if inspect.isclass(obj):
            return self._document_class_summary(obj, name, module_path)
        elif inspect.isfunction(obj) or callable(obj):
            return self._document_function(obj, name, module_path)
        elif not inspect.ismodule(obj):
            # Document as a constant/attribute
            return {
                "@type": "PropertyValue",
                "name": name,
                "value": str(type(obj).__name__),
                "description": f"Constant or attribute of type {type(obj).__name__}"
            }
        return None
    
    def _document_class_summary(self, cls: type, name: str, module_path: str) -> Dict:
        """Create a summary entry for a class."""
        doc = inspect.getdoc(cls) or ""
        parsed = parse_docstring(doc) if doc else None
        
        return {
            "@type": "Class",
            "@id": f"pxt:{module_path}.{name}",  # Use full path as stable identifier
            "name": name,
            "description": parsed.short_description if parsed else "",
            "url": f"{self.base_url}/{module_path.replace('.', '/')}/{name}"
        }
    
    def _document_function(self, func: Any, name: str, module_path: str) -> Dict:
        """Document a function."""
        doc = inspect.getdoc(func) or ""
        parsed = parse_docstring(doc) if doc else None
        
        # Get source location
        source_info = self._get_source_location(func)
        
        func_entry = {
            "@type": "Function",
            "@id": f"pxt:{module_path}.{name}",  # Stable identifier using module path
            "name": f"{module_path}.{name}",  # Full qualified name
            "identifier": name,  # Just the function name
            "description": parsed.short_description if parsed else "",
            "memberOf": {
                "@type": "SoftwareSourceCode",
                "@id": f"pxt:{module_path}"
            }
        }
        
        # Add GitHub URL if available
        if source_info and source_info.get("@id"):
            func_entry["sameAs"] = source_info["@id"]  # GitHub URL with line number
            func_entry["url"] = source_info["@id"]  # Also use as primary URL
        
        # Add additional source info if available
        if source_info:
            func_entry["codeLocation"] = {
                "startLine": source_info.get("line"),
                "lineCount": source_info.get("lineCount"),
                "relativePath": source_info.get("relativePath")
            }
        
        # Add signature
        try:
            sig = inspect.signature(func)
            func_entry["signature"] = self._format_signature_json(sig)
            
            # Add additional semantic metadata
            func_entry["potentialAction"] = {
                "@type": "InvokeAction",
                "target": {
                    "@type": "EntryPoint",
                    "urlTemplate": func_entry["@id"],
                    "httpMethod": "GET"
                }
            }
            
            # Infer function category/intent from name and context
            func_entry["category"] = self._infer_category(name, module_path)
            
            # Add usage frequency hint (could be calculated from codebase analysis)
            if name in ['create_table', 'select', 'where', 'insert']:
                func_entry["usageFrequency"] = "high"
            
        except:
            func_entry["signature"] = {"parameters": [], "returns": "Any"}
        
        # Add parameters
        if parsed and parsed.params:
            func_entry["parameters"] = [
                {
                    "@type": "PropertyValue",
                    "name": param.arg_name,
                    "valueType": param.type_name or "Any",
                    "description": param.description or ""
                }
                for param in parsed.params
            ]
        
        # Add return type
        if parsed and parsed.returns:
            func_entry["returns"] = {
                "@type": "PropertyValue",
                "valueType": parsed.returns.type_name or "Any",
                "description": parsed.returns.description or ""
            }
        
        return func_entry
    
    def _document_method(self, method: Any, name: str, class_name: str) -> Dict:
        """Document a method."""
        doc = inspect.getdoc(method) or ""
        parsed = parse_docstring(doc) if doc else None
        
        method_entry = {
            "@type": "Method",
            "@id": f"pxt:{class_name}.{name}",  # Use class.method as stable identifier
            "name": name,
            "description": parsed.short_description if parsed else "",
            "memberOf": {
                "@type": "Class",
                "@id": f"pxt:{class_name}"
            }
        }
        
        # Add source location
        source_info = self._get_source_location(method)
        if source_info:
            method_entry["sourceLocation"] = source_info
        
        # Add signature
        try:
            sig = inspect.signature(method)
            method_entry["signature"] = self._format_signature_json(sig)
        except:
            method_entry["signature"] = {"parameters": [], "returns": "Any"}
        
        # Add parameters (excluding self)
        if parsed and parsed.params:
            method_entry["parameters"] = [
                {
                    "@type": "PropertyValue",
                    "name": param.arg_name,
                    "valueType": param.type_name or "Any",
                    "description": param.description or ""
                }
                for param in parsed.params
                if param.arg_name != 'self'
            ]
        
        return method_entry
    
    def _format_signature_json(self, sig: inspect.Signature) -> Dict:
        """Format signature as JSON structure with formatted signature string."""
        params = []
        param_strings = []
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            param_info = {
                "name": param_name,
                "required": param.default == inspect.Parameter.empty,
            }
            
            # Build parameter string for formatted signature
            param_str = param_name
            
            if param.annotation != inspect.Parameter.empty:
                param_info["type"] = str(param.annotation)
                param_str += f": {param.annotation}"
            
            if param.default != inspect.Parameter.empty:
                param_info["default"] = str(param.default)
                if param.default is None:
                    param_str += " = None"
                elif isinstance(param.default, str):
                    param_str += f" = '{param.default}'"
                else:
                    param_str += f" = {param.default}"
            
            params.append(param_info)
            param_strings.append(param_str)
        
        result = {
            "parameters": params,
            "formatted": self._wrap_signature_params(param_strings)
        }
        
        if sig.return_annotation != inspect.Signature.empty:
            result["returns"] = str(sig.return_annotation)
        
        return result
    
    def _wrap_signature_params(self, param_strings: List[str]) -> str:
        """Format parameters with line breaks after commas for readability."""
        if not param_strings:
            return "()"
        
        # ALWAYS wrap after commas for consistency
        wrapped = "(\n    " + ",\n    ".join(param_strings) + "\n)"
        return wrapped
    
    def _infer_category(self, name: str, module_path: str) -> str:
        """Infer the category/intent of a function from its name and module."""
        name_lower = name.lower()
        
        # Check module context first
        if 'functions.image' in module_path:
            return "image_processing"
        elif 'functions.video' in module_path:
            return "video_processing"
        elif 'functions.audio' in module_path:
            return "audio_processing"
        elif 'integrations' in module_path:
            return "external_integration"
        
        # Check function name patterns
        if name_lower.startswith('create_') or name_lower.startswith('init_'):
            return "initialization"
        elif name_lower.startswith('get_') or name_lower.startswith('fetch_'):
            return "retrieval"
        elif name_lower.startswith('update_') or name_lower.startswith('set_'):
            return "modification"
        elif name_lower.startswith('delete_') or name_lower.startswith('drop_'):
            return "deletion"
        elif name_lower.startswith('validate_') or name_lower.startswith('check_'):
            return "validation"
        elif name_lower.startswith('convert_') or name_lower.startswith('to_'):
            return "transformation"
        elif 'query' in name_lower or 'select' in name_lower:
            return "query"
        elif 'insert' in name_lower or 'add' in name_lower:
            return "data_ingestion"
        else:
            return "utility"
    
    def _get_source_location(self, obj: Any) -> Optional[Dict]:
        """Get source location information for an object."""
        try:
            source_file = inspect.getsourcefile(obj)
            if not source_file:
                return None
            
            source_lines, line_number = inspect.getsourcelines(obj)
            source_path = Path(source_file)
            
            # Try to get GitHub link (primary identifier)
            path_parts = source_path.parts
            github_url = None
            relative_path = None
            
            for i, part in enumerate(path_parts):
                if part == 'pixeltable':
                    relative_path = '/'.join(path_parts[i:])
                    github_url = f"https://github.com/pixeltable/pixeltable/blob/{self.version}/{relative_path}#L{line_number}"
                    break
            
            if not github_url:
                # Fallback to local path if can't construct GitHub URL
                return {
                    "@id": f"file://{source_path}#L{line_number}",
                    "localPath": str(source_path),
                    "line": line_number,
                    "lineCount": len(source_lines)
                }
            
            # Use GitHub URL as the primary @id
            location = {
                "@id": github_url,  # Primary identifier is GitHub URL
                "@type": "SoftwareSourceCode",
                "url": github_url,
                "line": line_number,
                "lineCount": len(source_lines),
                "relativePath": relative_path,
                "localPath": str(source_path)  # Keep local for reference
            }
            
            return location
        except (TypeError, OSError):
            return None
    
    def _create_error_entry(self, path: str, error: str, item_type: str) -> Dict:
        """Create an error entry for items that couldn't be loaded."""
        return {
            "@type": "Error",
            "name": path,
            "description": f"Failed to load {item_type}: {error}",
            "error": True
        }
    
    def save(self, output_file: Optional[Path] = None, flatten: bool = True):
        """Save the LLM map to a JSON-LD file.
        
        Args:
            output_file: Path to save the file (defaults to mintlifier/llm_map.jsonld)
            flatten: If True and pyld is available, also save a flattened version
        """
        if not output_file:
            # Save in mintlifier directory, not the output directory
            mintlifier_dir = Path(__file__).parent
            output_file = mintlifier_dir / "llm_map.jsonld"
        
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the regular hierarchical version
        with open(output_file, 'w') as f:
            json.dump(self.api_map, f, indent=2)
        
        print(f"📊 LLM map saved to: {output_file}")
        
        # Save flattened version if requested and pyld is available
        if flatten and HAS_PYLD:
            try:
                # Flatten the document
                flattened = jsonld.flatten(self.api_map)
                
                # Save flattened version
                flattened_file = output_file.with_stem(output_file.stem + "_flat")
                with open(flattened_file, 'w') as f:
                    json.dump(flattened, f, indent=2)
                
                print(f"📊 Flattened LLM map saved to: {flattened_file}")
                
                # Also try to create a compacted version with a simpler context
                context = {
                    "pxt": "https://pixeltable.com/ontology#",
                    "schema": "https://schema.org/",
                    "name": "schema:name",
                    "description": "schema:description",
                    "signature": "pxt:signature",
                    "parameters": "pxt:parameters",
                    "category": "pxt:category",
                    "codeLocation": "pxt:codeLocation",
                    "memberOf": "schema:memberOf"
                }
                
                compacted = jsonld.compact(self.api_map, context)
                compacted_file = output_file.with_stem(output_file.stem + "_compact")
                with open(compacted_file, 'w') as f:
                    json.dump(compacted, f, indent=2)
                
                print(f"📊 Compacted LLM map saved to: {compacted_file}")
                
            except Exception as e:
                print(f"⚠️ Could not create flattened/compacted versions: {e}")
    
    def add_to_map(self, entry: Dict):
        """Add an entry to the main API map."""
        if entry and not entry.get("error"):
            self.api_map["hasPart"].append(entry)