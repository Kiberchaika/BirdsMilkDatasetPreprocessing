import os
import concurrent.futures
from webdav3.client import Client
from datetime import datetime
from tqdm import tqdm

# Параметры
WEBDAV_URL = 'http://81.94.156.17:5050'
USERNAME = 'webdavuser'
PASSWORD = 'webdavpassword'
TARGET_DIR = '/media/k4_nas/disk2/Music'
THREADS = 10

'''
sudo apt install apache2 apache2-utils

chmod +x setup-webdav.sh
sudo ./setup-webdav.sh
---

#!/bin/bash

# WebDAV directory
WEBDAV_DIR="/mnt/DATASET_FOLDER"

# Ensure directories exist
sudo mkdir -p "$WEBDAV_DIR"
sudo mkdir -p /var/lib/dav

# Set permissions
# sudo chown -R www-data:www-data "$WEBDAV_DIR"
# sudo chown -R www-data:www-data /var/lib/dav
# sudo chmod -R 775 "$WEBDAV_DIR" 

# Enable required modules
sudo a2enmod dav
sudo a2enmod dav_fs
sudo a2enmod headers

# firewall
sudo ufw allow ssh
sudo ufw allow 5050/tcp
sudo ufw enable

# Add port to Apache's ports.conf file
sudo tee -a /etc/apache2/ports.conf <<EOF
Listen 5050
EOF

# Create configuration file
sudo tee /etc/apache2/sites-available/webdav.conf <<EOF
DavLockDB /var/lib/dav/lockdb

<VirtualHost *:5050>
    ServerAdmin webmaster@localhost
    DocumentRoot /mnt/DATASET_FOLDER

    # CORS Headers
    Header always set Access-Control-Allow-Origin "*"
    Header always set Access-Control-Allow-Headers "Authorization,Content-Type,Depth"
    Header always set Access-Control-Allow-Methods "GET,HEAD,PUT,DELETE,OPTIONS,PROPFIND,PROPPATCH,COPY,MOVE,LOCK,UNLOCK"

    <Directory /mnt/DATASET_FOLDER>
        Options Indexes FollowSymLinks
        AllowOverride None
        Dav On
        
        AuthType Basic
        AuthName "WebDAV Storage"
        AuthUserFile /etc/apache2/.htpasswd
        Require valid-user

        <LimitExcept GET HEAD OPTIONS PROPFIND>
            Require valid-user
        </LimitExcept>
    </Directory>

    ErrorLog \${APACHE_LOG_DIR}/webdav_error.log
    CustomLog \${APACHE_LOG_DIR}/webdav_access.log combined
</VirtualHost>
EOF

# Create user and password
sudo htpasswd -cb /etc/apache2/.htpasswd webdavuser webdavpassword

# Test configuration
echo "Testing Apache configuration..."
if ! apache2ctl configtest; then
    echo "Apache configuration test failed"
    exit 1
fi

# Enable site and restart Apache
sudo a2dissite 000-default.conf
sudo a2ensite webdav.conf

# Restart Apache
echo "Restarting Apache..."
sudo systemctl restart apache2

# Check status
sleep 2
if systemctl is-active --quiet apache2; then
    echo "Apache successfully restarted"
    echo "WebDAV server is running at http://$(hostname -I | awk '{print $1}')"
else
    echo "Apache failed to start. Checking error log:"
    sudo systemctl status apache2
    tail -n 20 /var/log/apache2/error.log
fi
'''


# Настройка клиента WebDAV
options = {
    'webdav_hostname': WEBDAV_URL,
    'webdav_login': USERNAME,
    'webdav_password': PASSWORD
}
client = Client(options)

def should_download(remote_path, local_path):
    if not os.path.exists(local_path):
        return True
    
    try:
        local_size = os.path.getsize(local_path)
        local_mtime = os.path.getmtime(local_path)
        
        remote_info = client.info(remote_path)
        remote_size = int(remote_info.get('size', 0))
        remote_mtime = datetime.strptime(remote_info.get('modified', ''), '%a, %d %b %Y %H:%M:%S GMT').timestamp()
        
        return local_size != remote_size# or local_mtime != remote_mtime
    except Exception as e:
        print(f"Ошибка проверки файла {remote_path}: {e}")
        return True

def download_file(args):
    remote_path, local_path = args
    try:
        if should_download(remote_path, local_path):
            # Get remote modification time before download
            remote_info = client.info(remote_path)
            remote_mtime = datetime.strptime(remote_info.get('modified', ''), '%a, %d %b %Y %H:%M:%S GMT').timestamp()
            
            # Download the file
            client.download_sync(remote_path=remote_path, local_path=local_path)
            
            # Set the local file's modification time to match remote
            #os.utime(local_path, (remote_mtime, remote_mtime))
            
            return f"Скачан {remote_path}"
        return f"Пропущен {remote_path} (актуален)"
    except Exception as e:
        return f"Ошибка при скачивании {remote_path}: {e}"

def normalize_path(path):
    # Убираем повторяющиеся слеши
    path = '/'.join(filter(None, path.split('/')))
    if path:
        path = '/' + path
    return path

def list_files_recursive(remote_path, processed_paths=None):
    if processed_paths is None:
        processed_paths = set()
    
    files = []
    try:
        # Нормализуем текущий путь
        remote_path = normalize_path(remote_path)
        
        # Проверяем, не обрабатывали ли мы уже этот путь
        if remote_path in processed_paths:
            return files
        
        processed_paths.add(remote_path)
        print(f"Listing directory: {remote_path}")
        
        items = client.list(remote_path)
        for item in items:
            # Удаляем trailing slash
            item = item.rstrip('/')
            
            # Формируем полный путь
            full_path = normalize_path(f"{remote_path}/{item}")
            
            # Проверяем является ли элемент директорией по наличию trailing slash в исходном имени
            if item + '/' in items:
                # Рекурсивно обрабатываем поддиректории
                files.extend(list_files_recursive(full_path, processed_paths))
            else:
                files.append(full_path)
                
    except Exception as e:
        print(f"Ошибка при получении списка файлов из {remote_path}: {e}")
    
    return files

# Получение списка всех файлов
print("Получение списка файлов...")
all_files = list_files_recursive('/')
total_files = len(all_files)
print(f"Найдено файлов: {total_files}")

# Подготовка списка задач
download_tasks = []
for remote_file in all_files:
    # Создаем корректный локальный путь
    local_file = os.path.join(TARGET_DIR, remote_file.lstrip('/'))
    os.makedirs(os.path.dirname(local_file), exist_ok=True)
    download_tasks.append((remote_file, local_file))

# Скачивание файлов
with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as executor:
    results = list(tqdm(
        executor.map(download_file, download_tasks),
        total=total_files,
        desc="Скачивание файлов",
        unit="файл"
    ))

# Вывод статистики
success = sum(1 for r in results if "Скачан" in r)
skipped = sum(1 for r in results if "Пропущен" in r)
errors = sum(1 for r in results if "Ошибка" in r)
print("\nСтатистика:")
print(f"Успешно скачано: {success}")
print(f"Пропущено: {skipped}")
print(f"Ошибок: {errors}")
print("Скачивание завершено.")