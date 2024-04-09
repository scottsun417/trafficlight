*压缩包里是已经训练好的模型(.ckpt)还不是可以在线服务的(.pb)，待解决
	---[已解决] 可以用ckpt直接测试, 具体代码在myservice.zip

*重新训练步骤
	*删除"data/cache", "output/res101/*", "data/VOCdevkit2007/annotations_cache"
	*cd data/coco/PythonAPI/ 	make
	*cd lib	make clean, make
	*回到根目录(tf-faster-rcnn-master)，
		".experiments/scripts/train_faster_rcnn.sh 0 pascal_voc res101"
	
*从https://github.com/endernewton/tf-faster-rcnn开始测试可能会遇到的问题
	1.无法git clone：建议从网站上clone，传到obs，再导入notebook
	2.notebook里测试，无需修改GPU的型号
	3.没有权限创建soft link：直接把解压后的文件夹重命名为"VOCdevkit2007”，该文件夹应该包括"VOC2007","VOCcode"等文件夹。
	4.pre_trained model下载不了：从网站下载，传到obs，导入notebook
	======== 训练自己的模型========
	5.cd lib/datasets
	6.修改pascal_voc.py里的类，切勿删除__background__
	7.修改pascal_voc.py里function _load_pascal_annotation，x1,y1,x2,y2的"-1"，否则会出现loss=NAN. (注意xml文件里对bbox的定义，如果使用的不是x1,y1,x2,y2，而是x,y,w,h，这里要对应修改，因不同数据集而异)
	8.修改imdb.py, line25, num_classed=自己的类总数+1, (垃圾分类44类+1=45)
	9.修改imbd.py, line116,117, 去掉"-1", 否则loss会出现NAN
	10.修改voc_eval.py, line121, open(cachefile, 'b')改为'wb' (py2用b, py3用wb)
	11.替换dataset，直接把VOCdevkit2007/VOC2007里的Annotation，JPEGImage，ImageSets三个文件夹换掉。
	12.ImageSets/Main里应该有"trainval.txt", "test.txt"，可以打开我写的genDatasets.ipynb，运行一下就能生成，可以修改其中的比例和路径


遇到其他问题，可以随时问我，一起讨论解决。