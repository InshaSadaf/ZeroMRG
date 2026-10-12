"""Focused metadata parser tests; no authentication/network/model operations."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from gate4_execution_control import parse_dataset_metadata

FIXTURE=dict(datasetId=123,datasetSlug='dataset',ownerUser='inshasadaf114',title='COV-CTR',isPrivate=True)

class MetadataParserTests(unittest.TestCase):
    def test_dictionary(self): self.assertEqual(parse_dataset_metadata(FIXTURE),FIXTURE)
    def test_valid_json_string(self): self.assertEqual(parse_dataset_metadata(json.dumps(FIXTURE)),FIXTURE)
    def test_malformed_json(self):
        with self.assertRaisesRegex(ValueError,'malformed JSON'): parse_dataset_metadata('{invalid')
    def test_missing_required_keys(self):
        for key in FIXTURE:
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'missing required key'):
                parse_dataset_metadata({k:v for k,v in FIXTURE.items() if k!=key})
    def test_unexpected_data_types(self):
        for value in (None,[],5,True,'[]','null','"string"',{'info':None}):
            with self.subTest(value=value),self.assertRaises(ValueError):parse_dataset_metadata(value)
        for key,value in (('datasetId',True),('datasetId','123'),('datasetId',0),('isPrivate','true'),
                          ('datasetSlug',[]),('ownerUser',None),('title','')):
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                parse_dataset_metadata(dict(FIXTURE,**{key:value}))
    def test_existing_sdk_metadata_structure(self):
        from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetMetadataResponse,DatasetInfo
        info=DatasetInfo();info.dataset_id=123;info.dataset_slug='dataset';info.owner_user='inshasadaf114'
        info.title='COV-CTR';info.is_private=True
        response=ApiGetDatasetMetadataResponse();response.info=info
        old_serialization=response.to_json(response.info)
        self.assertIsInstance(old_serialization,str)
        self.assertIn('datasetId',json.loads(old_serialization))
        self.assertEqual(parse_dataset_metadata(old_serialization)['datasetId'],123)
        self.assertEqual(parse_dataset_metadata(response.to_json(info,ignore_defaults=False))['isPrivate'],True)
        info.is_private=False
        self.assertFalse(parse_dataset_metadata(response.to_json(info,ignore_defaults=False))['isPrivate'])
    def test_response_error_rejected(self):
        for envelope in ({'info':FIXTURE,'errorMessage':'denied'},{'info':FIXTURE,'errorMessage':None}):
            with self.assertRaisesRegex(ValueError,'invalid/error'):parse_dataset_metadata(envelope)

if __name__=='__main__':unittest.main(verbosity=2)
