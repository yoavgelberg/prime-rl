import json

import pytest
import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import GenerationConfig, PreTrainedTokenizerFast

from .hero_exports import file_hash, verify


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def export_files(tmp_path):
    revision = '768f209d9ea81521153ed38c47d515654e938aea'
    reference, policy = tmp_path / revision, tmp_path / 'policy-0016'
    reference.mkdir(); policy.mkdir()
    config = dict(model_type='qwen3', hidden_size=4, vocab_size=8, tie_word_embeddings=True,
        rope_theta=1000000., max_position_embeddings=262144, bos_token_id=1, eos_token_id=3)
    save(reference / 'config.json', config)
    save(policy / 'config.json', config)
    GenerationConfig(eos_token_id=[3, 4], bos_token_id=1, pad_token_id=0).save_pretrained(reference)
    vocab = {word: i for i, word in enumerate(['[PAD]', '[BOS]', '[UNK]', '[EOS]', 'a', 'b', 'c', 'd'])}
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(vocab, unk_token='[UNK]')),
        pad_token='[PAD]', bos_token='[BOS]', unk_token='[UNK]', eos_token='[EOS]')
    tokenizer.chat_template = '{% for message in messages %}{{message.role}}: {{message.content}}{% endfor %}'
    tokenizer.save_pretrained(reference); tokenizer.save_pretrained(policy)
    generator = torch.Generator().manual_seed(17)
    embedding = torch.randn(8, 4, generator=generator)
    weights = {'model.embed_tokens.weight': embedding, 'lm_head.weight': embedding,
        'model.layers.0.weight': torch.randn(4, 4, generator=generator)}
    identity = dict(variant='hero-v2', reference_checkpoint=str(reference), model_revision=revision,
        source_sha256='a' * 64, window=1024)
    checkpoint = tmp_path / 'latest.pt'
    torch.save(dict(format='ps-hero-v2-adamw-v1', model=weights, policy_version=16, identity=identity), checkpoint)
    save(checkpoint.with_suffix('.pt.json'), dict(policy_version=16, identity=identity, bytes=checkpoint.stat().st_size))
    save(policy / 'experiment.json', dict(identity, policy_version=16))
    save_file({key: value.bfloat16() for key, value in weights.items() if key != 'lm_head.weight'},
        str(policy / 'model.safetensors'))
    return checkpoint, policy, tmp_path / 'catalog/policy-0016.json'


def test_export_verifies_every_effective_weight_and_supplies_original_generation_config(export_files):
    checkpoint, policy, output = export_files
    before = checkpoint.read_bytes()
    receipt = verify(checkpoint, policy, output, expected_parameters=48)
    assert receipt['status'] == 'VERIFIED' and receipt['unique_parameters'] == 48
    assert receipt['policy_version'] == 16 and len(receipt['tensors']) == 2
    assert receipt['generation_config_supplied_from_reference'] is True
    assert json.loads((policy / 'generation_config.json').read_text())['eos_token_id'] == [3, 4]
    assert checkpoint.read_bytes() == before
    assert json.loads(output.read_text()) == receipt
    for row in receipt['files']:
        assert file_hash(policy / row['name']) == row['sha256']
    assert verify(checkpoint, policy, output, expected_parameters=48)['generation_config_supplied_from_reference'] is False


def test_export_rejects_one_corrupted_effective_weight(export_files):
    checkpoint, policy, output = export_files
    path = policy / 'model.safetensors'
    tensors = load_file(path)
    tensors['model.layers.0.weight'][0, 0] += 1
    save_file(tensors, path)
    with pytest.raises(ValueError, match='effective BF16 training weights'):
        verify(checkpoint, policy, output, expected_parameters=48)
    assert not output.exists()


@pytest.mark.parametrize('target,changes,message', [
    ('experiment.json', dict(policy_version=32), 'policy versions differ'),
    ('experiment.json', dict(window=512), 'training identities differ'),
    ('config.json', dict(rope_theta=10000.), 'architecture, RoPE'),
    ('generation_config.json', dict(eos_token_id=7), 'generation configuration differs'),
])
def test_export_rejects_wrong_policy_or_inference_configuration(export_files, target, changes, message):
    checkpoint, policy, output = export_files
    path = policy / target
    save(path, (json.loads(path.read_text()) if path.exists() else {}) | changes)
    with pytest.raises(ValueError, match=message):
        verify(checkpoint, policy, output, expected_parameters=48)


def test_export_rejects_changed_chat_template(export_files):
    checkpoint, policy, output = export_files
    (policy / 'chat_template.jinja').write_text('Changed template')
    with pytest.raises(ValueError, match='native chat template changed'):
        verify(checkpoint, policy, output, expected_parameters=48)


def test_export_requires_external_receipt_and_exact_parameter_count(export_files):
    checkpoint, policy, output = export_files
    with pytest.raises(ValueError, match='outside the export directory'):
        verify(checkpoint, policy, policy / 'manifest.json', expected_parameters=48)
    with pytest.raises(ValueError, match='parameter count differs'):
        verify(checkpoint, policy, output, expected_parameters=49)
