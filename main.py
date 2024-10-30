import pdb
import src
import glob
import os
import cv2
import importlib.machinery

path = 'Images{}*'.format(os.sep)
all_submissions = glob.glob('./src/*')
os.makedirs('./results/', exist_ok=True)

for idx, algo in enumerate(all_submissions):
    print('****************\tRunning Awesome Stitcher developed by: {}  | {} of {}\t********************'.format(
        algo.split(os.sep)[-1], idx, len(all_submissions)))
    try:
        module_name = '{}_{}'.format(algo.split(os.sep)[-1], 'stitcher')
        filepath = '{}{}stitcher.py'.format(algo, os.sep)
        
        # Use importlib.machinery instead of importlib.util
        loader = importlib.machinery.SourceFileLoader(module_name, filepath)
        module = loader.load_module()
        
        PanaromaStitcher = getattr(module, 'PanaromaStitcher')
        inst = PanaromaStitcher()

        for impaths in glob.glob(path):
            print('\t\t Processing... {}'.format(impaths))
            stitched_image, homography_matrix_list = inst.make_panaroma_for_images_in(path=impaths)

            outfile = './results/{}/{}.png'.format(impaths.split(os.sep)[-1], module_name)
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            cv2.imwrite(outfile, stitched_image)
            print(homography_matrix_list)
            print('Panaroma saved ... @ ./results/{}.png'.format(module_name))
            print('\n\n')

    except Exception as e:
        print('Oh No! My implementation encountered this issue\n\t{}'.format(e))
        print('\n\n')