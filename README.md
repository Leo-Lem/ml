# Machine Learning Project for NLP

### Train Spacy NER model using the GermaNER dataset. 
---
### Project can be done in group (2-4 students). There will be presentation <span style="color:red">4/6 December</span>. You need to prepare a small report (<span style="color:red">1-2 pages</span>) and a presentation (<span style="color:red">max 5 minutes</span>). All of the group members should participate in the presentation.
---

## Assignment details

1. Train a new spaCy NER model for German (blank) using the GermaNER datasets - see <span style="color:red">Assignment-Option-1 folder</span>.
2. Load the existing German NER model from spaCy and update the model using the GermaNER dataset. Make sure to normalize entity types accordingly. For example, if the built-in entity type for person is <span style="color:blue">PERSON</span> but it is marked as <span style="color:blue">PER</span> in the GermaNER dataset, convert the GermaNER lable to <span style="color:blue">PERSON</span>. There are also some special "derivative" and "part" lables such as <span style="color:blue">ORGpart</span> and <span style="color:blue">ORGderiv</span>. How it will affect the the performance? if you normalize these types to a common one, example ORGpart and ORGderiv to <span style="color:blue">ORG</span>?
3. Train and update the model using the train and dev dataset and test it with test dataset. Use the perl script provided to evaluate the performance. 
<span style="color:red">The test file will be provided on December 2</span>

**Note**: You can ignore the last column from training and testing! For the evaluation purpose, you can re-use the gold labels.
 


## TSV Format

The following snippet shows an example of the TSV format we use in this task. 

```tsv
1 Aufgrund O O 
2 seiner O O
3 Initiative O O
4 fand O O
5 2001/2002 O O
6 in O O
7 Stuttgart B-LOC O
8 , O O
9 Braunschweig B-LOC O
10 und O O
11 Bonn B-LOC O
12 eine O O
13 große O O
14 und O O
15 publizistisch O O
16 vielbeachtete O O
17 Troia-Ausstellung B-LOCpart Ov
18 statt O O
19 , O O
20 „ O O
21 Troia B-OTH B-LOC
22 - I-OTH O
23 Traum I-OTH O
24 und I-OTH O
25 Wirklichkeit I-OTH O
26 “ O O
27 . O O
```



## Evaluation with perl script

You have to use the provided scripts for the evaluation of your results. <span style="color:red">The Perl scripts can be executed for the NER classifier file as:</span>

```sh
perl conlleval_ner.pl < your_classified_file
```

For executing the evaluation script, you need to have perl 5 installed. Copy the evaluation script and the file to be evaluated (here: `eval-sample-d.tsv`) in the same directory.

**Note**: For the file to be evaluated, you can keep the outer column all as "O".

```sh
perl nereval.perl < eval-sample-d.tsv
> STRICT: Found: 10 outer and 5 inner phrases; Gold: 12 (outer) and 1 (inner).
> LOOSE: Found: 10 outer and 5 inner phrases; Gold: 12 (outer) and 1 (inner).
> 
> 1. Strict, Combined Evaluation (official):
> Accuracy:  92.47%;
> Precision:  40.00%;
> Recall:  46.15%;
> FB1:  42.86
> 
> 2. Loose, Combined Evaluation:
> Accuracy:  93.84%;
> Precision:  53.33%;
> Recall:  61.54%;
> FB1:  57.14
> 
> 3.1 Per-Level Evaluation (outer chunks):
> Accuracy:  91.78%;
> Precision:  60.00%;
> Recall:  50.00%;
> FB1:  54.55
> 
> 3.2 Per-Level Global Evaluation (inner chunks):
> Accuracy:  93.15%;
> Precision:   0.00%;
> Recall:   0.00%;
> FB1:   0.00
```