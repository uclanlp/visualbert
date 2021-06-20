import imagehash
import json
import os
import progressbar
import signal
import socket
import sys
import requests

from PIL import Image

json_file = sys.argv[1]
save_dir = sys.argv[2]

split_name = json_file.split(".")[0]

HEADER = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
TIMEOUT = 2 # Timeout of 2 sections. 

examples = [json.loads(line) for line in open(json_file).readlines()]

class Timeout():
    """Timeout class using ALARM signal."""
    class Timeout(Exception):
        pass

    def __init__(self, sec):
        self.sec = sec

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.raise_timeout)
        signal.alarm(self.sec)

    def __exit__(self, *args):
        signal.alarm(0)    # disable alarm

    def raise_timeout(self, *args):
        raise Timeout.Timeout()

def save_image(filename, url, img_hash, wrong_hash_file):
    save_path = os.path.join(save_dir, filename)
    if not os.path.exists(save_path):
        try:
            with Timeout(TIMEOUT):
                try:
                    request = requests.get(url, headers = HEADER, stream = True)

                    # Save the image to the specified directory
                    with open(save_path, 'wb') as f:
                        for chunk in request.iter_content(1024):
                            f.write(chunk)

                    # And make sure the hash is correct
                    try:
                        saved_hash = str(imagehash.average_hash(Image.open(save_path)))

                        if not saved_hash == img_hash:
                            wrong_hash_file.write(str(url) + "\t" + str(filename) + "\t" + str(saved_hash) + "\t" + str(img_hash) + "\n")
                    except OSError as e:
                        return e
                except requests.exceptions.ConnectionError as e:
                    return e
                except requests.exceptions.TooManyRedirects as e:
                    return e
                except requests.exceptions.ChunkedEncodingError as e:
                    return e
                except requests.exceptions.ContentDecodingError as e:
                    return e
                return request.status_code
        except Timeout.Timeout as e:
            return e

pbar = progressbar.ProgressBar(maxval=len(examples))

hash_file = sys.argv[3]
hashes = json.loads(open(hash_file).read())

pbar.start()
with open(split_name + "_failed_imgs.txt", "a") as ofile, open(split_name + "_checked_imgs.txt", "a") as checked_file, open(split_name + "_failed_hashes.txt", "a") as failed_hash_file:
    checked_urls = set([line.strip() for line in open(split_name + "_checked_imgs.txt").readlines()])
    num_none = 0
    num_total = 0

    for i, example in enumerate(examples):
        split_id = example["identifier"].split("-")
        image_id = "-".join(split_id[:3])

        left_image_name = image_id + "-img0.png"
        right_image_name = image_id + "-img1.png"

        left_url = example["left_url"]
        right_url = example["right_url"]

        if not left_url in checked_urls:
            status_code = save_image(left_image_name, left_url, hashes[left_image_name], failed_hash_file)
            if status_code != 200:
                ofile.write(str(status_code) + "\t" + left_image_name + "\t" + left_url + "\n")
                ofile.flush()
                num_none += 1
            checked_urls.add(left_url)
            checked_file.write(left_url + "\n")
            num_total += 1
        if not right_url in checked_urls:
            status_code = save_image(right_image_name, right_url, hashes[right_image_name], failed_hash_file)
            if status_code != 200:
                ofile.write(str(status_code) + "\t" + right_image_name + "\t" + right_url + "\n")
                ofile.flush()
                num_none += 1
            checked_urls.add(right_url)
            checked_file.write(right_url + "\n")
            num_total += 1

        pbar.update(i)

pbar.finish()
print("number of missing images: " + str(num_none))
print("total number of requests: " + str(num_total))
