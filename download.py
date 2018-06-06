from google.cloud import storage
client = storage.Client()
bucket = client.get_bucket('spk_bucket1')
blob = storage.Blob('2data_log.txt', bucket)
#content = blob.download_as_string()
#print(content)

with open('/home/sujithpk/Desktop/d.csv', 'wb') as file_obj:
    blob.download_to_file(file_obj)
