"""Small CPU filesystem fixtures; no encoders, reports scored, or remote access."""
import copy
import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from zeromrg_cov_data_resolver import (ATTACHMENT, CovDataError, file_hash, ids_hash,
    resolve_cov_dataset, validate_contract, verify_checkpoint_manifests)


class ResolverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root/'input'; self.input.mkdir()
        self.mount = self.input/'datasets/inshasadaf114/dataset'
        self.mount.mkdir(parents=True)
        self.data = self.mount/'COV-CTR with English reports/COV-CTR'
        self.data.mkdir(parents=True)
        self.names = [f'{i}.png' for i in range(746)]
        self.content = {i: ('image fixture '+i).encode() for i in self.names}
        self.csv = b'image_id,reports_En\nfixture,not a clinical report\n'
        (self.data/'reports_ZH_EN.csv').write_bytes(self.csv)
        for i, content in self.content.items(): (self.data/i).write_bytes(content)
        groups = dict(validated=self.names[:714], train=self.names[:571],
                      validation=self.names[571:642], test=self.names[642:714], prompt=self.names[:500])
        self.contract = dict(schema='zeromrg_cov_identity_v1', attachment=ATTACHMENT,
            archive_filename='COV-CTR with English reports.zip',
            annotation=dict(filename='reports_ZH_EN.csv', sha256=hashlib.sha256(self.csv).hexdigest()),
            splits=groups, split_ids_sha256={n:ids_hash(ids) for n,ids in groups.items()},
            images={i:dict(bytes=len(c), sha256=hashlib.sha256(c).hexdigest()) for i,c in self.content.items()},
            physical_image_count=746, prompt_ids_sha256=ids_hash(groups['prompt']),
            manifest_sha256={'train_manifest.csv':hashlib.sha256(b'pinned manifest').hexdigest()})

    def resolve(self, contract=None):
        c = contract or self.contract
        return resolve_cov_dataset(self.input, self.mount/'COV-CTR with English reports', c,
            self.contract['splits']['train'], self.contract['splits']['prompt'], self.root/'scratch')

    def fails(self, code, function=None):
        with self.assertRaises(CovDataError) as caught: (function or self.resolve)()
        self.assertEqual(caught.exception.diagnostic['code'],code)
        json.dumps(caught.exception.diagnostic)

    def test_expected_mounted_layout(self):
        result = self.resolve()
        self.assertEqual(result['root'],self.data.resolve())
        self.assertEqual(result['evidence']['valid_pairs'],714)
        self.assertFalse(result['evidence']['report_text_parsed'])

    def test_short_slug_and_nested_layout(self):
        target = self.input/'dataset/wrapper/extra/COV-CTR'
        target.parent.mkdir(parents=True)
        self.data.rename(target)
        result = self.resolve()
        self.assertEqual(result['root'],target.resolve())
        self.assertEqual(Path(result['evidence']['attachment_root']), (self.input/'dataset').resolve())

    def test_missing_image(self):
        (self.data/'700.png').unlink()
        self.fails('IMAGE_INVENTORY_MISMATCH')

    def test_extra_image(self):
        (self.data/'unapproved.png').write_bytes(b'extra')
        self.fails('IMAGE_INVENTORY_MISMATCH')

    def test_ambiguous_dataset_roots(self):
        other = self.mount/'other/COV-CTR';other.mkdir(parents=True)
        (other/'reports_ZH_EN.csv').write_bytes(self.csv)
        self.fails('DATASET_ROOT_MISSING_OR_AMBIGUOUS')

    def test_incorrect_raw_manifest(self):
        (self.data/'reports_ZH_EN.csv').write_bytes(self.csv+b'changed')
        self.fails('ANNOTATION_CONTENT_MISMATCH')

    def test_wrong_checkpoint_manifest(self):
        path = self.root/'checkpoint.zip'
        with zipfile.ZipFile(path,'w') as z: z.writestr('dependencies/train_manifest.csv',b'wrong')
        def verify():
            with zipfile.ZipFile(path) as z: verify_checkpoint_manifests(z,self.contract)
        self.fails('CHECKPOINT_MANIFEST_HASH_MISMATCH',verify)

    def test_checkpoint_manifest_correct_and_missing(self):
        path = self.root/'checkpoint.zip'
        with zipfile.ZipFile(path,'w') as z: z.writestr('dependencies/train_manifest.csv',b'pinned manifest')
        with zipfile.ZipFile(path) as z: verify_checkpoint_manifests(z,self.contract)
        with zipfile.ZipFile(path,'w'): pass
        def verify():
            with zipfile.ZipFile(path) as z: verify_checkpoint_manifests(z,self.contract)
        self.fails('CHECKPOINT_MANIFEST_MISSING_OR_AMBIGUOUS',verify)

    def test_mismatched_split_membership(self):
        c=copy.deepcopy(self.contract)
        c['splits']['test'][0]=c['splits']['train'][0]
        self.fails('SPLIT_MEMBERSHIP_MISMATCH',lambda:self.resolve(c))

    def test_changed_training_order(self):
        c=copy.deepcopy(self.contract);c['splits']['train'].reverse()
        self.fails('TRAIN_ORDER_OR_MEMBERSHIP_MISMATCH',lambda:self.resolve(c))

    def test_validation_test_swap_preserving_counts_rejected(self):
        c=copy.deepcopy(self.contract)
        c['splits']['validation'][0],c['splits']['test'][0]=c['splits']['test'][0],c['splits']['validation'][0]
        self.fails('SPLIT_ID_FINGERPRINT_MISMATCH',lambda:self.resolve(c))

    def test_equivalent_input_path(self):
        result=resolve_cov_dataset(self.input/'.', self.data/'..', self.contract,
            self.contract['splits']['train'],self.contract['splits']['prompt'],self.root/'scratch')
        self.assertEqual(result['root'], self.data.resolve())

    def test_prompt_bank_mismatch_and_order(self):
        for replacement in (self.names[500],self.names[1]):
            c=copy.deepcopy(self.contract);c['splits']['prompt'][0]=replacement
            with self.assertRaises(CovDataError): self.resolve(c)
        c=copy.deepcopy(self.contract);c['splits']['prompt'].reverse()
        self.fails('PROMPT_BANK_MEMBERSHIP_OR_ORDER_MISMATCH',lambda:self.resolve(c))

    def test_genuine_image_content_mismatch(self):
        (self.data/'10.png').write_bytes(b'changed image')
        self.fails('IMAGE_CONTENT_MISMATCH')

    def test_duplicate_image_identifier(self):
        nested=self.data/'nested';nested.mkdir();(nested/'10.png').write_bytes(self.content['10.png'])
        self.fails('DUPLICATE_IMAGE_IDENTIFIER')

    def test_unapproved_attachment_even_with_same_files(self):
        destination=self.input/'wrong-dataset';self.mount.rename(destination)
        self.fails('APPROVED_ATTACHMENT_NOT_FOUND')

    def test_attachment_metadata_mismatch(self):
        c=copy.deepcopy(self.contract);c['attachment']['dataset_id']=123
        self.fails('ATTACHMENT_CONTRACT_MISMATCH',lambda:self.resolve(c))

    def create_archive(self, extra=None):
        path=self.mount/self.contract['archive_filename']
        with zipfile.ZipFile(path,'w') as z:
            z.writestr('nested/COV-CTR/reports_ZH_EN.csv',self.csv)
            for i,c in self.content.items(): z.writestr('nested/COV-CTR/'+i,c)
            if extra: z.writestr(extra,b'unsafe')
        # Keep no second extracted candidate; relocate the fixture outside input.
        self.data.rename(self.root/'original_fixture')
        return path

    def test_original_archive_nested_layout_crc_and_extraction(self):
        self.create_archive()
        result=self.resolve()
        self.assertEqual(result['evidence']['kind'],'archive')
        self.assertEqual(len(result['images']),714)
        self.assertEqual(file_hash(result['images']['0.png']),self.contract['images']['0.png']['sha256'])
        self.assertFalse((result['root']/'714.png').exists())

    def test_archive_unsafe_member(self):
        self.create_archive('../outside')
        self.fails('UNSAFE_ARCHIVE_MEMBER')

    def test_archive_and_extracted_layout_are_ambiguous(self):
        path=self.mount/self.contract['archive_filename']
        with zipfile.ZipFile(path,'w') as z: z.writestr('COV-CTR/reports_ZH_EN.csv',self.csv)
        self.fails('DATASET_ROOT_MISSING_OR_AMBIGUOUS')

    def test_filesystem_order_does_not_change_mapping(self):
        result=self.resolve()
        self.assertEqual(result['images']['10.png'].name,'10.png')
        self.assertEqual(result['evidence']['prompt_ids_sha256'],ids_hash(self.names[:500]))


if __name__=='__main__': unittest.main(verbosity=2)
