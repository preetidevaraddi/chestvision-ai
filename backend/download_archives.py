import requests
import os
import sys
import zipfile

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    print(f"Fetching token for {id}...")
    response = session.get(URL, params={'id': id}, stream=True)

    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    else:
        # Sometimes small files don't require confirmation token
        pass

    if response.status_code != 200:
        print(f"Failed to download. Status code: {response.status_code}")
        return False

    print(f"Downloading to {destination}...")
    save_response_content(response, destination)
    return True

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def download_and_verify():
    files = {
        "dataset/raw/LungCancer.zip": "1QpiK8OwLljLv13MK7VHKoey-5Mhplap1",
        "dataset/raw/Normal.zip": "1HIonSVtcllI3jeK6uz2FGGlMN9N4hxeI",
        "dataset/raw/Pneumonia.zip": "162tNzvyNcewx4ajCzlzXqyHy8kRoqg-y",
        "dataset/raw/Tuberculosis.zip": "13A7Pw7kVkvsNT7IZ4e_eSeQ8t4GxIU17"
    }

    os.makedirs("dataset/raw", exist_ok=True)

    for dest, file_id in files.items():
        if not os.path.exists(dest):
            print(f"\n--- Processing {dest} ---")
            success = download_file_from_google_drive(file_id, dest)
            if not success:
                print(f"ERROR: Failed to download {dest}.")
                return False
        else:
            print(f"\nAlready have {dest}")
            
        if not os.path.exists(dest):
            print(f"ERROR: Failed to download {dest}.")
            return False

        # Verify ZIP integrity
        print(f"Verifying {dest}...")
        try:
            with zipfile.ZipFile(dest, 'r') as zip_ref:
                bad_file = zip_ref.testzip()
                if bad_file:
                    print(f"ERROR: {dest} is corrupted (bad file: {bad_file})")
                    return False
                print(f"SUCCESS: {dest} downloaded and verified. Size: {os.path.getsize(dest) / (1024*1024):.2f} MB")
        except zipfile.BadZipFile:
            print(f"ERROR: {dest} is not a valid ZIP file.")
            return False
            
    return True

if __name__ == "__main__":
    download_and_verify()
