# Download fashion mnist dataset
# and save it to the specified directory
import os
import requests
import gzip
import shutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def download_fashion_mnist(data_dir):
    