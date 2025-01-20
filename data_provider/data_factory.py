from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader
from data_provider.uea import collate_fn
from data_provider.batch_stride_sampler import StrideSampler
from torch.utils.data import DataLoader, BatchSampler

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader
}

def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    
    if args.batch_not_shuffle == True:
        if flag == 'test' or flag == 'TEST':
            shuffle_flag = False
        elif flag == 'train' or flag == 'TRAIN':
            shuffle_flag = False
        else:
            shuffle_flag = True
    else:    
        shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    if args.sample_stride == True:
        stride = 16
    else:
        stride = 1

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader

    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns,
            stride=stride
        )
        print(flag, len(data_set))

        if args.sample_stride == True:
            sampler = StrideSampler(args, data_set, stride)
            batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=drop_last)

            data_loader = DataLoader(
                data_set,
                #batch_size=batch_size,
                #shuffle=shuffle_flag,
                num_workers=args.num_workers,
                #drop_last=drop_last,
                batch_sampler=batch_sampler,
            )
            return data_set, data_loader
        else:
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last)
        return data_set, data_loader
