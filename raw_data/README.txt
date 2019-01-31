MS SEG challenge training dataset

Version 3

Adds consensus segmentations made from manual segmentations using logarithmic opinion pool based STAPLE. These segmentations will be used for evaluation. Also re-ordered manual segmentations so that the numbers correspond to the same expert for each patient.

----------

Version 2

Corrects some errors in manual segmentation orientation matrices for two patients (07040 and 07043)

----------

Version 1

This file contains the unprocessed images for the 15 patients of the challenge training dataset. These patients are coming from three different MRI scanners located in three different centers (see https://portal.fli-iam.irisa.fr/msseg-challenge/data for more details on the centers and scanners). Patients starting with 01 come from a Siemens 3T Verio scanner, those starting with 07 from a Siemens Aera 1.5T scanner, and finally those starting with 08 from a Philips Ingenia 3T scanner.

The provided images include, for each patient:

- 3D FLAIR image
- 3D T1 image
- T2 image
- DP image
- 3D T1 Gd image (post-contrast agent image)
- seven manual segmentations from clinical experts: binary maps in the referential of the 3D FLAIR image.

No preprocessing was applied to any of the provided images. Another set of images with default pre-processing is provided in another file. Consensus segmentations that will be used for evaluation will be provided soon.

For any question or remark on the images provided, please contact challenges-iam@inria.fr.
