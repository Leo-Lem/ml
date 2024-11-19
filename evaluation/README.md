For executing the evaluation script, you need to have perl 5 installed. Copy the evaluation script and the file to be evaluated (here: eval-sample-d.tsv)  in the same directory, and execute:

For the file to be evaluated, you can keep the outer column all as "O"


shell>  perl nereval.perl < eval-sample-d.tsv 

For this sample file, evaluation results look as follows:

STRICT: Found: 10 outer and 5 inner phrases; Gold: 12 (outer) and 1 (inner).
LOOSE: Found: 10 outer and 5 inner phrases; Gold: 12 (outer) and 1 (inner).

1. Strict, Combined Evaluation (official):
Accuracy:  92.47%;
Precision:  40.00%;
Recall:  46.15%;
FB1:  42.86

2. Loose, Combined Evaluation:
Accuracy:  93.84%;
Precision:  53.33%;
Recall:  61.54%;
FB1:  57.14

3.1 Per-Level Evaluation (outer chunks):
Accuracy:  91.78%;
Precision:  60.00%;
Recall:  50.00%;
FB1:  54.55

3.2 Per-Level Global Evaluation (inner chunks):
Accuracy:  93.15%;
Precision:   0.00%;
Recall:   0.00%;
FB1:   0.00
