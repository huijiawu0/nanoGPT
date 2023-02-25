import os
import sys
import boto3

ACCESS_KEY = 'AKIA5AKOSQ7KIVCK7CJZ'
SECRET_KEY = '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'
BUCKET_NAME = sys.argv[1]
FILE_NAME = sys.argv[2]
# KEY_PREFIX = 'wudao'

# Set the chunk size for each part
PART_SIZE = 100 * 1024 * 1024

# Create an S3 client with your credentials
s3 = boto3.client(
    's3',
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY
)

# Create a multipart upload request
mpu_response = s3.create_multipart_upload(
    Bucket=BUCKET_NAME,
    Key=f'{os.path.basename(FILE_NAME)}'
)

# Get the upload ID
upload_id = mpu_response['UploadId']

# Open the file to be uploaded
with open(FILE_NAME, 'rb') as f:
    # Initialize variables
    part_number = 0
    parts = []

    while True:
        # Read the next part
        data = f.read(PART_SIZE)
        part_number += 1

        # If no more data, we're done
        if not data:
            break

        # Upload the part
        response = s3.upload_part(
            Body=data,
            Bucket=BUCKET_NAME,
            Key=f'{os.path.basename(FILE_NAME)}',
            UploadId=upload_id,
            PartNumber=part_number
        )

        # Record the ETag so we can reference the part when completing the upload
        parts.append({
            'PartNumber': part_number,
            'ETag': response['ETag']
        })

# Complete the multipart upload
s3.complete_multipart_upload(
    Bucket=BUCKET_NAME,
    Key=f'{os.path.basename(FILE_NAME)}',
    UploadId=upload_id,
    MultipartUpload={
        'Parts': parts
    }
)

print('Upload completed successfully.')
