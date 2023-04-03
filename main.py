import argparse
from segment.predict import mainfile
from segment.u_net_test import foto_predict, video_predict

def run(source, weights):
    source, weights = opt.source, opt.weights

    if source.endswith(".png") or source.endswith(".jpg"):
        if weights.endswith(".h5") or weights.endswith(".hdf5"):
            foto_predict(source, weights)

        if weights.endswith(".pt"):
            mainfile(source, weights)
        
    if source.endswith(".mp4"):
        if weights.endswith(".h5") or weights.endswith(".hdf5"):
            video_predict(source,weights)

        if weights.endswith(".pt"):
            mainfile(source,weights)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='model path(s)')
    parser.add_argument('--source', type=str, help='file/dir/URL/glob, 0 for webcam')
    opt = parser.parse_args()
    return opt

def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
