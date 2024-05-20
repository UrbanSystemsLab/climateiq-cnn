set -e

mkdir -p "$(dirname $0)/files/cloud_function_source"
rsync "$(dirname $0)/../usl_pipeline/cloud_functions/main.py" "$(dirname $0)/files/cloud_function_source/"
rsync "$(dirname $0)/../usl_pipeline/cloud_functions/requirements.txt" "$(dirname $0)/files/cloud_function_source/"
rsync -r --include '*.py' --exclude '*.pyc' "$(dirname $0)/../usl_pipeline/usl_lib/usl_lib" "$(dirname $0)/files/cloud_function_source/"
