import os
import click

@click.command()
@click.option('--ont', '-ont', help="Ontology (bp, cc or mf)")

def main(ont):

  # Extracting embeddings
  print('Extracting embeddings...')
  for model_name in ['esm', 't5']:
    print('model_name')
    os.system(f'python3 scripts/extract.py -m {model_name} -ont {ont}')
  print('Extract - Done')

  # DIAMOND
  print('DIAMOND...')
  os.system(f'python3 scripts/diamond.py -ont {ont}')
  print('DIAMOND - Done')

  # Training - MLP
  print('MLP training...')
  for layer in ['36', '35', '34', '24', '23', '22']:
    os.system(f'python3 scripts/mlp.py -l {layer} -ont {ont}')
  print('MLP training - Done')

  # Training - ResNet50
  print('ResNet50 training...')
  os.system(f'python3 scripts/resnet50.py -ont {ont}')
  print('ResNet50 training - Done')

  # Stacking
  print('Stacking...')
  os.system(f'python3 scripts/stacking.py -ont {ont}')
  print('Stacking - Done')

  # Ensemble
  print('Ensemble...')
  os.system(f'python3 scripts/ensemble.py -ont {ont}')
  print('Ensemble - Done')


if __name__ == '__main__':
  main()