#!/bin/bash

cd op_importance

# python2 workflow_classes/Workflow.py 'svm_maxmin' 'average' 'True' 'True'
python2 workflow_classes/Workflow.py 'svm_maxmin' 'average' 'False' 'True'
# python2 workflow_classes/Workflow.py 'svm_maxmin' 'average' 'False' 'False'

# python2 workflow_classes/Workflow.py 'svm_maxmin' 'complete' 'False' 'True'
# python2 workflow_classes/Workflow.py 'svm_maxmin' 'complete' 'False' 'False'