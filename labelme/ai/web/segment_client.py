import requests
import threading
import collections
from ...logger import logger
from ...utils import img_arr_to_b64
import imgviz
import numpy as np

class SegmentClient:
    def __init__(self, url):
        self._enc_url = url["encoder"]["url"]
        self._dec_url = url["decoder"]["url"]
        self._lock = threading.Lock()
        self._image_embedding_cache = collections.OrderedDict()

    def set_image(self, image):
        with self._lock:
            self._image = image
            self._image_embedding = self._image_embedding_cache.get(
                self._image.tobytes()
            )
        self._thread = threading.Thread(
            target=self._compute_and_cache_image_embedding
        )
        self._thread.start()

    def _compute_and_cache_image_embedding(self):
        with self._lock:
            logger.debug("Computing image embedding...")
            self._image_embedding = self._request_enc(self._image, self._enc_url)
            if len(self._image_embedding_cache) > 10:
                self._image_embedding_cache.popitem(last=False)
            self._image_embedding_cache[
                self._image.tobytes()
            ] = self._image_embedding
            logger.debug("Done computing image embedding.")

    def _request_enc(self, image, url):
        image = imgviz.asrgb(image)
        inputs = {
            "image": img_arr_to_b64(image)
        }
        results = requests.post(url, data=inputs)
        return np.array(results.json()['results'])
    
    def _request_dec(self, embedding, points, point_labels, url):
        inputs = {
            "embedding": str(embedding),
            "image_shape": str(self._image.shape[:2]),
            "point": str(points),
            "point_labels": str(point_labels)
        }
        results = requests.post(url, data=inputs)

        return eval(results.json()['results'])
    
    def _get_image_embedding(self):
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        with self._lock:
            return self._image_embedding
        
    def _predict_polygon_from_points(self, points, point_labels):
        image_embedding = self._get_image_embedding()
        points = self._request_dec(image_embedding.tolist(), points, point_labels, self._dec_url)
        print('_predict_polygon_from_points')
        return points