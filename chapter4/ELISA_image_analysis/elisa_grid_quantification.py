#!/usr/bin/env python3
"""Quantify grayscale ELISA/antibody-array spots from a manually gridded image.

The input image MUST already contain a complete red grid that defines the assay
wells. This script does not infer spot centers and does not draw the grid.

The default settings reproduce the analysis used for the example 8 x 11 array:
- local background: lower-left 2 x 2 pixels of each grid cell
- local high reference: mean of the 10 darkest pixels in inverted grayscale
- signal threshold: background + 0.30 * (high reference - background)
- reported primary signal: mean inverted-gray value of pixels above threshold
"""
from __future__ import annotations
import argparse
from pathlib import Path
import string
import cv2
import numpy as np
import pandas as pd

def parse_args():
    p=argparse.ArgumentParser(description='Quantify a manually red-gridded ELISA/antibody-array image.')
    p.add_argument('image', type=Path)
    p.add_argument('--rows', type=int, default=8)
    p.add_argument('--cols', type=int, default=11)
    p.add_argument('--output-dir', type=Path, default=Path('output'))
    p.add_argument('--prefix', default=None)
    p.add_argument('--background-patch', type=int, default=2)
    p.add_argument('--top-pixels', type=int, default=10)
    p.add_argument('--signal-fraction', type=float, default=0.30)
    p.add_argument('--grid-line-fraction', type=float, default=0.50)
    p.add_argument('--red-min', type=int, default=180)
    p.add_argument('--green-max', type=int, default=80)
    p.add_argument('--blue-max', type=int, default=80)
    return p.parse_args()

def contiguous_runs(indices):
    if indices.size==0: return []
    breaks=np.where(np.diff(indices)>1)[0]
    starts=np.r_[0, breaks+1]; ends=np.r_[breaks, len(indices)-1]
    return [(int(indices[s]),int(indices[e])) for s,e in zip(starts, ends)]

def detect_grid_runs(red_mask, axis, minimum_fraction):
    occupancy=red_mask.mean(axis=axis)
    return contiguous_runs(np.where(occupancy>=minimum_fraction)[0])

def row_name(index):
    result=''; n=index+1
    while n:
        n,rem=divmod(n-1,26); result=string.ascii_uppercase[rem]+result
    return result

def main():
    a=parse_args()
    if a.rows<=0 or a.cols<=0: raise ValueError('--rows and --cols must be positive')
    if not 0<=a.signal_fraction<=1: raise ValueError('--signal-fraction must be between 0 and 1')
    img=cv2.imread(str(a.image.resolve()), cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(a.image)
    h,w=img.shape[:2]
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_inv=255.0-gray.astype(np.float32)
    b,g,r=cv2.split(img)
    red_mask=(r>=a.red_min)&(g<=a.green_max)&(b<=a.blue_max)
    x_runs=detect_grid_runs(red_mask, axis=0, minimum_fraction=a.grid_line_fraction)
    y_runs=detect_grid_runs(red_mask, axis=1, minimum_fraction=a.grid_line_fraction)
    if len(x_runs)!=a.cols+1 or len(y_runs)!=a.rows+1:
        raise RuntimeError(f'Expected {a.cols+1} vertical and {a.rows+1} horizontal grid lines; detected {len(x_runs)} and {len(y_runs)}. The red grid must be drawn manually and completely.')
    background_color=np.array([235,206,135],dtype=np.uint8)
    low_signal_color=np.array([0,255,0],dtype=np.uint8)
    classified=img.copy(); results=[]
    for row in range(a.rows):
        for col in range(a.cols):
            x1=x_runs[col][1]+1; x2=x_runs[col+1][0]
            y1=y_runs[row][1]+1; y2=y_runs[row+1][0]
            cell=gray_inv[y1:y2,x1:x2]; ch,cw=cell.shape
            patch=min(a.background_patch,ch,cw)
            bg_mask=np.zeros_like(cell,dtype=bool); bg_mask[ch-patch:ch,0:patch]=True
            bg=float(cell[bg_mask].mean())
            vals=cell.ravel(); n_top=min(a.top_pixels, vals.size)
            high=float(np.partition(vals, vals.size-n_top)[-n_top:].mean())
            threshold=bg+a.signal_fraction*max(0.0, high-bg)
            usable=~bg_mask
            bgpix=usable&(cell<=bg); low=usable&(cell>bg)&(cell<threshold); sig=usable&(cell>=threshold)
            out=classified[y1:y2,x1:x2]; out[bgpix]=background_color; out[low]=low_signal_color
            sv=cell[sig]; usable_count=int(usable.sum())
            results.append({'well':f'{row_name(row)}{col+1}','row':row+1,'column':col+1,
                            'background_value':bg,'top_pixel_mean':high,'signal_threshold':float(threshold),
                            'signal_mean':float(sv.mean()) if sv.size else 0.0,
                            'signal_mean_bg_subtracted':float((sv-bg).mean()) if sv.size else 0.0,
                            'signal_sum_bg_subtracted':float((sv-bg).sum()) if sv.size else 0.0,
                            'signal_pixels':int(sv.size),'usable_pixels':usable_count,
                            'signal_pixel_fraction':float(sv.size/usable_count) if usable_count else 0.0})
    a.output_dir.mkdir(parents=True,exist_ok=True)
    prefix=a.prefix or a.image.stem
    out_img=a.output_dir/f'{prefix}_classified.png'; out_xlsx=a.output_dir/f'{prefix}_signals.xlsx'
    cv2.imwrite(str(out_img),classified)
    params=pd.DataFrame([{'parameter':'input_image','value':str(a.image)}, {'parameter':'rows','value':a.rows}, {'parameter':'cols','value':a.cols}, {'parameter':'background_patch_px','value':a.background_patch}, {'parameter':'top_pixels','value':a.top_pixels}, {'parameter':'signal_fraction','value':a.signal_fraction}, {'parameter':'grid_line_fraction','value':a.grid_line_fraction}, {'parameter':'red_min','value':a.red_min}, {'parameter':'green_max','value':a.green_max}, {'parameter':'blue_max','value':a.blue_max}, {'parameter':'image_width_px','value':w}, {'parameter':'image_height_px','value':h}])
    with pd.ExcelWriter(out_xlsx,engine='openpyxl') as wr:
        pd.DataFrame(results).to_excel(wr,sheet_name='signals',index=False); params.to_excel(wr,sheet_name='parameters',index=False)
    print(out_img); print(out_xlsx)
if __name__=='__main__': main()
