cat "$(dirname "$0")/map_uploader/requirements.txt" "$(dirname "$0")/cloud_functions/requirements.txt" \
    | sed "s/..\/usl_lib/usl_lib/" \
    | LC_COLLATE=C sort \
    | uniq \
	  > "$(dirname "$0")/requirements.txt"
