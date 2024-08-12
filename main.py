import util
from run import train, test


if __name__=='__main__':
    # parse options and make directories
    argv = util.option.parse()

    if argv.train:
        train(argv)
    if argv.test:
        test(argv)
        
    exit(0)
