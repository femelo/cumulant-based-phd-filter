# Labeled Cumulant-based PHD filter

This is a README file for a Python implementation of a labeled extension of the cumulant-based PHD filter also known as [linear complexity filter](https://ieeexplore.ieee.org/document/8455331) for tracking objects in images.

## Licensing
This code is licensed by Flávio Eler De Melo under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License (CC-BY-NC-SA)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). A copy of the legal code for CC-BY-NC-SA public license can be found in the [LICENSE.md](LICENSE.md) file.

By exercising the Licensed Rights (defined in the [Creative Commons BY-NC-SA Legal Code](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)), You accept and agree to be bound by the terms and conditions of this Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License ("Public License"), as stated in the [Creative Commons BY-NC-SA Legal Code](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). To the extent this Public License may be interpreted as a contract, You are granted the Licensed Rights in consideration of Your acceptance of these terms and conditions, and the Licensor grants You such rights in consideration of benefits the Licensor receives from making the Licensed Material available under these terms and conditions stated at the [Creative Commons BY-NC-SA Legal Code](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

**Note:** No data is provided with the code due to licensing or lack thereof.

## Contact
Please get in touch by messages for addressing doubts or suggestions. If you have found a bug, please open a ticket.

## Requirements
  - Ubuntu >= 16.04
  - Python 2 or 3
  - OpenCV >= 3.2
  - Munkres == 1.0.12

## **1.** Installing system dependencies.

```
sudo apt-get install python python-pip python-dev python3-dev python3-pip libopencv-dev python-opencv python3-opencv imagemagick libtiff-tools
```

**Note:** The package *imagemagick* provides an image manipulation tools including *convert* and the package *libtiff-tools* provides the tools *tiffinfo* and *tiffcrop*. These tools may be needed to extract image sequences encapsulated in TIFF containers. The script 'Data/extract.sh' is provided for that purpose
and depends on those tools.

## **2.** Clone the repository.

```
git clone https://github.com/femelo/cumulant-based-phd-filter
```

## **3.** Install Python dependencies.

Change directory to the code base directory.
```
cd cumulant-based-phd-filter
```

For Python 2:
```
pip install -r requirements_python2.txt
```

For Python 3:
```
pip3 install -r requirements_python3.txt
```

## **4.** Download and untar the data file in the source directory.
```
tar zxvf Data.tar.gz
```
**Note:** No data is provided with the code due to licensing or lack thereof.

## **5.** Edit the *settings.json* file for changing the algorithm settings.

## **6.** Run the code.

For Python 2:
```
python track.py -ds WAMI -iv wami01 -d -v
```

For Python 3:
```
python3 track.py -ds WAMI -iv wami01 -d -v
```

All options for the command line program can be seen by checking the help as:
```
python3 track.py -h
```

To run the tracker for all video sequences of the WAMI dataset, the provided bash
script does the job:
```
./run_tracker.sh
```
