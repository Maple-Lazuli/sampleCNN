import argparse
import os
import torch

from ml_infrastructure.model import Model
from ml_infrastructure.manager import Manager

from data_utils.data_manager import MnistDataManager
from models.resnet import ResNet, ResBlock, ResBottleneckBlock

from models.lenet import Net as Lenet
from models.babycnn import Net as Baby


def main(flags):
    os.environ['CUDA_VISIBLE_DEVICES'] = flags.gpus

    dm = MnistDataManager(batch_size=flags.batch_size, training_noise=flags.training_noise, stats=flags.stats,
                          train=flags.train_csv, val=flags.val_csv, test=flags.test_csv).dm

    model1 = Model(net=ResNet(1, ResBlock, [2, 2, 2, 2], useBottleneck=False, outputs=10), name='resnet-18')
    model1.criterion = torch.nn.CrossEntropyLoss()

    model2 = Model(net=ResNet(1, ResBlock, [3, 4, 6, 3], useBottleneck=False, outputs=10), name='resnet-34')
    model2.criterion = torch.nn.CrossEntropyLoss()

    model3 = Model(net=ResNet(1, ResBottleneckBlock, [3, 4, 6, 3], useBottleneck=True, outputs=10), name='resnet-50')
    model3.criterion = torch.nn.CrossEntropyLoss()

    model4 = Model(net=Lenet(), name='lenet')
    model4.criterion = torch.nn.CrossEntropyLoss()

    model5 = Model(net=Baby(), name='baby')
    model5.criterion = torch.nn.CrossEntropyLoss()

    manager = Manager(models=[model1, model2, model3, model4, model5], data_manager=dm, epochs=1,
                      start_watcher_app=flags.start_watcher,
                      ip=flags.watcher_ip, port=flags.watcher_port, window_size=flags.window_size,
                      eval_rate=flags.eval_rate)

    manager.perform()
    manager.save_watcher_results(save_location='./results', save_name='Models.json')
    print("Finished training and archived performance json.")
    try:
        torch.cuda.empty_cache()
    except:
        print("could not free the gpu")
    if flags.stop_watcher_on_end:
        manager.shutdown_watcher()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpus', type=str,
                        default='0',
                        help='Convert the psd to grey scale.')

    parser.add_argument('--start-watcher', type=bool,
                        default=True,
                        help='Boolean to start a watcher')

    parser.add_argument('--stop-watcher-on-end', type=bool,
                        default=False,
                        help='Boolean to stop the watcher after training')

    parser.add_argument('--watcher-ip', type=str,
                        default='0.0.0.0',
                        help='The IP to use for the watcher')

    parser.add_argument('--watcher-port', type=int,
                        default=5124,
                        help='The port to use for the watcher')

    parser.add_argument('--training-noise', type=bool,
                        default=False,
                        help='Boolean to indicate whether to add noise during training')

    parser.add_argument('--batch-size', type=int,
                        default=1000,
                        help='The batch size to use when feeding data')

    parser.add_argument('--eval-rate', type=int,
                        default=3,
                        help='How often the model should be evaluated')

    parser.add_argument('--window-size', type=int,
                        default=20,
                        help='The window size to use for determining when to stop training.')

    parser.add_argument('--stats', type=str,
                        default='./data/stats.json',
                        help='The location of the stats.json file.')

    parser.add_argument('--train-csv', type=str,
                        default='./data/train.csv',
                        help='The location of the training csv.')

    parser.add_argument('--val-csv', type=str,
                        default='./data/val.csv',
                        help='The location of the val csv.')

    parser.add_argument('--test-csv', type=str,
                        default='./data/test.csv',
                        help='The location of the test csv.')

    parsed_flags, _ = parser.parse_known_args()

    main(parsed_flags)
