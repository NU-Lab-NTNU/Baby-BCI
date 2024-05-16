# How to run?
If you want to just blindly run everything, you need to:
- `python copy_data.py`
- `python raw_file_parser.py` (this needs renaming and will be renamed)
- `python preprocessing.py`
- `python Transformer.py`
- `python AgeClassifier.py`

The `AgeClassifier.py` file can optionally be run with a `--model-path` flag, which if supplied loads a model from the specified folder and runs it on the testing dataset. Otherwise, it will train, validate and test a new model which you will be prompted to save in the end.

## About the data - a guide for the uninitiated
All the annotated data is located in `\\felles.ansatt.ntnu.no\ntnu\su\ips\nullab\Analysis\EEG\Looming\Silje-Adelen\3. BCI Annotation (Silje-Adelen)\1. Infant VEP Annotations\1 )Annotated_Silje`. The file `1 Final_Annotations_Silje (3).xlsx` gives specifics on all the annotations - whether it's done and the number of peaks found in the raw file.

The way it's structured, each of the subfolders is divided into infants older than 7 months and younger than 7 months at the time of testing. For each infant, what's relevant for our analysis are the `.raw` file and the `.evt` file. The amazing Silje-Adelen has been so kind as to annotate all the peaks she saw in these files, and the annotations are a part of the .evt file. In the `Comnt` column, each occurence of `oz` or `pz` signifies a peak in the Pz or Oz electrode in the 10-10 system. A schematic of the 10-10 electrode system is shown below:

![10-10](EEG_10-10_system.svg)

Before getting to the actual step of training a classifier, the data needs to be copied from the file server, preprocessed, split into training, validation and testing datasets, and from these, specific features must be extracted.

## How to improve models?
While I was working on the classifier, I based my work on what was detected by the model trained by Swati and Vegard. A *big* setback to the model is that it only uses the occipital and parietal electrodes when training their model. See my pdf report from PSY8005 for a better explanation as to why, but in essence, one should aim to include both these and the prefrontal electrodes to make the model better.