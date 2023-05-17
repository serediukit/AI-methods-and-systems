Learn_NN_5L_(TrainDir=r'D:\git\AI-methods-and-systems\Train\\',
             ValidDir=r'D:\git\AI-methods-and-systems\Valid\\',
             RezDir=r'D:\git\AI-methods-and-systems\rez_dir\\',
             NN_Name='NN_L5', Epochs=5, window_size=12, windoe_fuction='hann')

TestNN_(NetName=r'D:\git\AI-methods-and-systems\rez_dir\NN_L5_Best.hdf5',
            SourceDir=r'D:\git\AI-methods-and-systems\Test\\',
            TargetFile=r'D:\git\AI-methods-and-systems\rez_dir\Test\NN_L5_rez',
            window_size=12)

TestNN_(NetName=r'D:\git\AI-methods-and-systems\rez_dir\NN_L5_Best.hdf5',
            SourceDir=r'D:\git\AI-methods-and-systems\Train\\',
            TargetFile=r'D:\git\AI-methods-and-systems\rez_dir\Train\NN_L5_rez',
            window_size=12)

TestNN_(NetName=r'D:\git\AI-methods-and-systems\rez_dir\NN_L5_Best.hdf5',
            SourceDir=r'D:\git\AI-methods-and-systems\Valid\\',
            TargetFile=r'D:\git\AI-methods-and-systems\rez_dir\Valid\NN_L5_rez',
            window_size=12)


# TestNN_(NetName=r'D:\git\AI-methods-and-systems\rez_dir\NN_L5_Best.hdf5',
#             SourceDir=r'D:\git\AI-methods-and-systems\Test\\',
#             TargetFile=r'D:\git\AI-methods-and-systems\rez_dir\Test\NN_L5_rez',
#             window_size=18)
#
# TestNN_(NetName=r'D:\git\AI-methods-and-systems\rez_dir\NN_L5_Best.hdf5',
#             SourceDir=r'D:\git\AI-methods-and-systems\Train\\',
#             TargetFile=r'D:\git\AI-methods-and-systems\rez_dir\Train\NN_L5_rez',
#             window_size=18)
#
# TestNN_(NetName=r'D:\git\AI-methods-and-systems\rez_dir\NN_L5_Best.hdf5',
#             SourceDir=r'D:\git\AI-methods-and-systems\Valid\\',
#             TargetFile=r'D:\git\AI-methods-and-systems\rez_dir\Valid\NN_L5_rez',
#             window_size=18)


# TestNN_(NetName=r'D:\git\AI-methods-and-systems\rez_dir\NN_L5_Best.hdf5',
#             SourceDir=r'D:\git\AI-methods-and-systems\Test\\',
#             TargetFile=r'D:\git\AI-methods-and-systems\rez_dir\Test\NN_L5_rez',
#             window_size=12)
#
# TestNN_(NetName=r'D:\git\AI-methods-and-systems\rez_dir\NN_L5_Best.hdf5',
#             SourceDir=r'D:\git\AI-methods-and-systems\Train\\',
#             TargetFile=r'D:\git\AI-methods-and-systems\rez_dir\Train\NN_L5_rez',
#             window_size=12)
#
# TestNN_(NetName=r'D:\git\AI-methods-and-systems\rez_dir\NN_L5_Best.hdf5',
#             SourceDir=r'D:\git\AI-methods-and-systems\Valid\\',
#             TargetFile=r'D:\git\AI-methods-and-systems\rez_dir\Valid\NN_L5_rez',
#             window_size=12)
