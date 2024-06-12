import os
import shutil
import importlib.util

def get_module_path(module_name):
    module_spec = importlib.util.find_spec(module_name)
    if module_spec is not None and module_spec.origin is not None:
        return os.path.dirname(module_spec.origin)
    else:
        raise Exception(f"Module {module_name} not found")

def create_backup(file_path):
    backup_path = file_path + '.bak'
    if not os.path.exists(backup_path):
        shutil.copy2(file_path, backup_path)
        print(f'Backup created at {backup_path}')
    else:
        print(f'Backup already exists at {backup_path}')

def apply_patch(src_path, dest_path):
    if os.path.exists(dest_path):
        create_backup(dest_path)
    shutil.copy2(src_path, dest_path)
    print(f'Copied {src_path} to {dest_path}')

def apply_patches(patches_dir):
    for root, dirs, files in os.walk(patches_dir):
        for file in files:
            src_path = os.path.join(root, file)
            module_name = os.path.relpath(root, patches_dir).replace(os.sep, '.')
            module_path = get_module_path(module_name)
            dest_path = os.path.join(module_path, file)
            apply_patch(src_path, dest_path)

if __name__ == '__main__':
    patches_dir = os.path.join(os.path.dirname(__file__), 'patches')
    apply_patches(patches_dir)
