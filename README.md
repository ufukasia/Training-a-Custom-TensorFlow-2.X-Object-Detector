Elbette, GitHub README.md dosyasının geri kalan kısımlarını, yalnızca Türkçe bilen birinin anlayabileceği şekilde ve kodun yürütme sonrasında ürettiği çıktıları orijinal dilinde bırakarak Türkçeye çevirdim.

# TensorFlow-2.X ile kendi nesne tanıma programını yaz
### TensorFlow-GPU ile kendi nesne tanıma programını nasıl eğitebilirsin bunu öğren ...

## İçerik Tablosu
1. [TensorFlow-GPU kurulumu](https://github.com/ufukasia/Training-a-Custom-TensorFlow-2.X-Object-Detector#TensorFlow-GPU-kurulumu)
2. [Anaconda kurulumu ve ayarları](https://github.com/ufukasia/Training-a-Custom-TensorFlow-2.X-Object-Detector#Anaconda-kurulumu-ve-ayarları)
3. [Veri kümesi toplama ve etiketleme](https://github.com/ufukasia/Training-a-Custom-TensorFlow-2.X-Object-Detector#gathering-and-labeling-our-dataset)
4. [Eğitim için gerekli ayarlamalar](https://github.com/ufukasia/Training-a-Custom-TensorFlow-2.X-Object-Detector#generating-training-data)
5. [Eğitim için klasör yapısı ve ayarları](https://github.com/ufukasia/Training-a-Custom-TensorFlow-2.X-Object-Detector#configuring-the-training-pipeline)
6. [Modelin eğitimi](https://github.com/ufukasia/Training-a-Custom-TensorFlow-2.X-Object-Detector#training-the-model)
7. [Çıkarım grafiğinin (Inference Graph) dışa aktarımı](https://github.com/ufukasia/Training-a-Custom-TensorFlow-2.X-Object-Detector#exporting-the-inference-graph)
8. [Modelimizin testi](https://github.com/ufukasia/Training-a-Custom-TensorFlow-2.X-Object-Detector#testing-out-the-finished-model)

Bu repoda kendi kedi ve köpek modelimi tanıyan bir model eğittim ve bu eğitimde etiketleme işlemlerini otomatik olarak nasıl yaptığımı size anlatmak istiyorum.

## Sistem Gereksinimleri
Bir TensorFlow modeli eğitmek zorunda kaldığınızda sisteminiz bu eğitimi desteklemek zorundadır ve verimliliğini belirleyecektir. Eğitilen model AMD 2700X işlemci, Nvidia 2060s ekran kartı ve 16GB RAM ile gerçekleşecektir. Windows üzerinde TensorFlow-GPU kütüphanesinin çalışması için Nvidia ekran kartı sahibi olmanız gerekmektedir. Eğer değilseniz Colab üzerinden Google Drive'a dosyalarınızı yükleyerek de sanal bir Tesla-V100 sahibi gibi eğitim yapabilirsiniz. Dersimiz ilgi görürse ayrıca bununla ilgili bir ders yapılacak. Siz de bu konu hakkında detaylı bilgi için bu linke bakabilirsiniz. [tıkla](https://developer.nvidia.com/cuda-gpus).

<p align="left">
  <img src="doc/cuda.png">
</p>
Uyumlu bir GPU'nuz olup olmadığından emin değilseniz, iki seçenek vardır. Birincisi deneme yanılma yapmaktır. CUDA Runtime'ı yükleyip ve sisteminizin uyumlu olup olmadığına bakabilirsiniz. CUDA Yükleyici, sistem uyumluluğunuzu belirleyen yerleşik bir sistem denetleyicisine sahiptir. İkinci seçenek ise TensorFlow CPU (temelde sadece düz TensorFlow) kullanabilirsiniz, ancak bu TensorFlow-GPU'dan önemli ölçüde daha yavaştır ama aynı şekilde çalışır. Bunu denemedim, ancak buna karar verirseniz, daha sonra TensorFlow CPU için bahsettiğim alternatif adımları izleyin. Ayrıca, Aygıt Yöneticisi'ni açıp Ekran Bağdaştırıcılarınızı kontrol ederek NVIDIA Sürücüleriniz olup olmadığını da kontrol edebilirsiniz. NVIDIA Sürücüleriniz varsa, sorun yok demektir. Yoksa ilk olarak güncel driverınızı yükleyin.

## Adımlar
### TensorFlow-GPU Kurulumu
İlk adım olarak Anaconda programının kurulu olduğunu varsayıyorum. Eğer Anaconda programı yoksa bunu nasıl yapacağınızı anlatan birçok YouTube videosu mevcut, bunlara bakabilirsiniz. Ben Udemy dersimde anlatıyorum.

TensorFlow-GPU kurulumu için gerçekten çok karmaşık yöntemler mevcut. Bunu size en basit yöntem ile kod yazmadan nasıl halledebilirsiniz bunu anlatmak istiyorum, zira bu iş bu kadar kolayken bunu anlatan hiçbir kaynağa denk gelmedim. TensorFlow-GPU için gereksinimler Anaconda, CUDA ve cuDNN'dir. Bunların versiyonlarının ekran kartınızın versiyonuna uyumlu olması gerektiği gibi çeşitli path ayarları yapmak gerekmektedir. Ama Anaconda bunu nasıl yapıyor?

Öncelikle [İndirme Sayfası](https://www.anaconda.com/products/individual)'na giderek Anaconda'yı kuralım. Buradan, 64-bit grafik yükleyiciyi indirin ve kurulumu tamamlamak için adımları izleyin. Bu tamamlandıktan sonra, Anaconda Navigator'ı yüklemiş olmalısınız, onu açın. Buraya geldikten sonra, bir komut istemi açın.
<p align="left">
  <img src="doc/anaconda.png">
</p>
Ardından, bu komutla bir sanal ortam oluşturun:

```
conda create -n tensorflow pip python=3.6
```

Ardından, ortamı şu şekilde etkinleştirin:

```
conda activate tensorflow
```
**Yeni bir Anaconda Terminali açtığınızda sanal ortamda olmayacağınızı unutmayın. Bu nedenle, yeni bir komut istemi açarsanız, sanal ortamı etkinleştirmek için yukarıdaki komutu kullandığınızdan emin olun**

Artık Anaconda Sanal Ortamımız kurulduğuna göre, CUDA ve cuDNN'yi kurabiliriz. TensorFlow CPU kullanmayı planlıyorsanız, bu adımı atlayabilir ve TensorFlow Kurulumuna geçebilirsiniz. Farklı bir TensorFlow sürümü kullanıyorsanız, test edilmiş yapılandırmalara [buradan](https://www.tensorflow.org/install/source#tested_build_configurations) göz atın. TensorFlow-GPU'yu kurma hakkında daha fazla bilgi için [TensorFlow web sitesine](https://www.tensorflow.org/install/gpu) bakın.

TensorFlow için gerekli olan doğru CUDA ve cuDNN sürümlerini artık bildiğinize göre, bunları NVIDIA Web Sitesinden yükleyebiliriz. TensorFlow 2.3.0 için [cuDNN 7.6.5](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/Production/10.1_20191031/cudnn-10.1-windows10-x64-v7.6.5.32.zip) ve [CUDA 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base)'i kullandım. Diğer sürümler için [CUDA Arşivi](https://developer.nvidia.com/cuda-toolkit-archive) ve [cuDNN Arşivi](https://developer.nvidia.com/rdp/cudnn-archive)'ne bakın. Her iki dosyayı da indirdikten sonra, CUDA Yükleyiciyi çalıştırın ve CUDA'yı kurmak için kurulum sihirbazını izleyin, MSBuild ve Visual Studio ile ilgili bazı çakışmalar olabilir, bunları MSBuild Araçları ile Visual Studio Community'nin en yeni sürümünü yükleyerek çözebilirsiniz. CUDA Araç Kitini başarıyla yükledikten sonra, nereye kurulduğunu bulun (benim için C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1 içindeydi). Ardından cuDNN kütüphanesinin içeriğini CUDA Klasörüne çıkarın.
<p align="left">
  <img src="doc/cudnn.png">
</p>
Bunu tamamladıktan sonra, TensorFlow-GPU (veya TensorFlow CPU) kurmak için gereken her şeye sahibiz. Bu yüzden Anaconda komut istemimize geri dönebilir ve aşağıdaki komutu verebiliriz:

```
pip install tensorflow-gpu
```

TensorFlow CPU kuruyorsanız, bunun yerine şunu kullanın:

```
pip install tensorflow
```

Kurulum tamamlandıktan sonra, her şeyin düzgün bir şekilde kurulup kurulmadığını kontrol etmek için aşağıdaki kodu kullanabiliriz:
```
python
>>> import tensorflow as tf
>>> print(tf.__version__)
```
Her şey düzgün bir şekilde kurulduysa, "2.3.0" mesajını veya hangi TensorFlow sürümüne sahipseniz onu almalısınız. Bu, TensorFlow'un çalışır durumda olduğu ve çalışma alanımızı kurmaya hazır olduğumuz anlamına gelir. Şimdi bir sonraki adıma geçebiliriz!
**İçe aktarmada bir hata varsa, [C++ Build Tools ile Visual Studio 2019](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community&rel=16)'u yüklemeniz gerekir.**

### Anaconda Kurulumu ve Ayarları
TensorFlow Object Detection API için, modelimizi eğitmek için izlememiz gereken belirli bir dizin yapısı vardır. İşlemi biraz daha kolaylaştırmak için, gerekli dosyaların çoğunu bu repoya ekledim.

İlk olarak, doğrudan C:'de bir klasör oluşturun ve "TensorFlow" olarak adlandırın. Klasörü nereye koymak istediğiniz size kalmış, ancak bu dizin yolunun komutlarla uyumlu olması için daha sonra gerekli olacağını unutmayın. Bu klasörü oluşturduktan sonra, Anaconda Komut İstemine geri dönün ve şu komutla klasöre geçin:

```
cd C:\TensorFlow
```
Buraya geldikten sonra, [TensorFlow models deposunu](https://github.com/tensorflow/models) şu komutla klonlamanız gerekecek:

```
git clone https://github.com/tensorflow/models.git
```
Bu, tüm dosyaları models adlı bir dizine klonlamalıdır. Bunu yaptıktan sonra, C:\TensorFlow içinde kalın ve [bu](https://github.com/armaanpriyadarshan/Training-a-Custom-TensorFlow-2.X-Object-Detector/archive/master.zip) repoyu bir .zip dosyası olarak indirin. Ardından, aşağıda vurgulanan iki dosyayı, workspace ve scripts'i doğrudan TensorFlow dizinine çıkarın.
<p align="left">
  <img src="doc/clone.png">
</p>

Ardından, dizin yapınız şöyle görünmelidir:

```
TensorFlow/
└─ models/
   ├─ community/
   ├─ official/
   ├─ orbit/
   ├─ research/
└─ scripts/
└─ workspace/
   ├─ training_demo/
```
Dizin yapısını kurduktan sonra, Object Detection API için önkoşulları yüklemeliyiz. İlk önce protobuf derleyicisini şu komutla yüklememiz gerekiyor:

```
conda install -c anaconda protobuf
```
Ardından, şu komutla TensorFlow\models\research dizinine geçmelisiniz:

```
cd models\research
```
Ardından, protoları şu komutla derleyin:

```
protoc object_detection\protos\*.proto --python_out=.
```
Bunu yaptıktan sonra, terminali kapatın ve yeni bir Anaconda komut istemi açın. Daha önce oluşturduğumuz sanal ortamı kullanıyorsanız, etkinleştirmek için aşağıdaki komutu kullanın:

```
conda activate tensorflow
```
TensorFlow 2 ile pycocotools, Object Detection API için bir bağımlılıktır. Windows Desteği ile yüklemek için şunu kullanın:

```
pip install cython
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```
**Kurulum talimatlarına göre Visual C++ 2015 derleme araçlarının kurulu ve yolunuzda olması gerektiğini unutmayın. Bu pakete sahip değilseniz, [buradan](https://go.microsoft.com/fwlink/?LinkId=691126) indirin.**

Şu komutla models\research dizinine geri dönün:

```
cd C:\TensorFlow\models\research
```

Buraya geldikten sonra, kurulum betiğini kopyalayın ve çalıştırın:

```
copy object_detection\packages\tf2\setup.py .
python -m pip install .
```
Herhangi bir hata varsa, bir sorun bildirin, ancak bunlar büyük olasılıkla pycocotools sorunlarıdır, yani kurulumunuz yanlıştır. Ancak her şey plana göre gittiyse, kurulumunuzu şu komutla test edebilirsiniz:

```
python object_detection\builders\model_builder_tf2_test.py
```
Şuna benzer bir çıktı almalısınız:

```
[       OK ] ModelBuilderTF2Test.test_create_ssd_models_from_config
[ RUN      ] ModelBuilderTF2Test.test_invalid_faster_rcnn_batchnorm_update
[       OK ] ModelBuilderTF2Test.test_invalid_faster_rcnn_batchnorm_update
[ RUN      ] ModelBuilderTF2Test.test_invalid_first_stage_nms_iou_threshold
[       OK ] ModelBuilderTF2Test.test_invalid_first_stage_nms_iou_threshold
[ RUN      ] ModelBuilderTF2Test.test_invalid_model_config_proto
[       OK ] ModelBuilderTF2Test.test_invalid_model_config_proto
[ RUN      ] ModelBuilderTF2Test.test_invalid_second_stage_batch_size
[       OK ] ModelBuilderTF2Test.test_invalid_second_stage_batch_size
[ RUN      ] ModelBuilderTF2Test.test_session
[  SKIPPED ] ModelBuilderTF2Test.test_session
[ RUN      ] ModelBuilderTF2Test.test_unknown_faster_rcnn_feature_extractor
[       OK ] ModelBuilderTF2Test.test_unknown_faster_rcnn_feature_extractor
[ RUN      ] ModelBuilderTF2Test.test_unknown_meta_architecture
[       OK ] ModelBuilderTF2Test.test_unknown_meta_architecture
[ RUN      ] ModelBuilderTF2Test.test_unknown_ssd_feature_extractor
[       OK ] ModelBuilderTF2Test.test_unknown_ssd_feature_extractor
----------------------------------------------------------------------
Ran 20 tests in 45.304s

OK (skipped=1)
```
Bu, Anaconda Dizin Yapısını ve TensorFlow Object Detection API'yı başarıyla kurduğumuz anlamına gelir. Artık nihayet veri kümemizi toplayabilir ve etiketleyebiliriz. O halde bir sonraki adıma geçelim!

### Veri Kümesi Toplama ve Etiketleme
TensorFlow Object Detection API hazır olduğuna göre, modelin eğitileceği ve test edileceği resimleri toplamalı ve etiketlemeliyiz. Şu andan itibaren ihtiyaç duyulacak tüm dosyalar ```workspace\training_demo``` dizininde bulunacaktır. Bu yüzden bir saniye ayırın, etrafa bakın ve dizinin yapısına alışın.

- ```annotations```: Bu, modelimiz için gereken tüm eğitim verilerimizi saklayacağımız yerdir. Bununla, eğitim hattı için gereken CSV ve RECORD dosyalarını kastediyorum. Ayrıca modelimiz için etiketleri içeren bir PBTXT dosyası da var. Kendi veri kümenizi eğitiyorsanız train.record ve test.record dosyalarını silebilirsiniz, ancak benim Hap Sınıflandırıcı modelimi eğitiyorsanız bunları tutabilirsiniz.
- ```exported-models```: Bu, bitmiş çıkarım grafiğimizi dışa aktarıp saklayacağımız çıktı klasörümüzdür.
- ```images```: Bu klasör, bir test ve train klasöründen oluşur. Burada, muhtemelen tahmin edebileceğiniz gibi, eğitim ve test için gereken etiketli görüntüleri saklayacağız. Etiketli görüntüler, orijinal görüntü ve bir XML dosyasından oluşur. Hap Sınıflandırıcı modelini eğitmek istiyorsanız, görüntüleri ve XML belgelerini tutabilirsiniz, aksi takdirde görüntüleri ve XML dosyalarını silin.
- ```models```: Bu klasörde, eğitim hattımızı ve eğitim işinden gelen kontrol noktası bilgilerini ve eğitim için gereken CONFIG dosyasını saklayacağız.
- ```pre-trained-models```: Burada, eğitim için başlangıç kontrol noktası olarak kullanacağımız önceden eğitilmiş modelimizi saklayacağız.
- Geri kalan betikler sadece modeli eğitmek ve dışa aktarmak için kullanılır, ayrıca bir test görüntüsü üzerinde çıkarım yapan örnek bir nesne algılama betiği de vardır.

Kendi özel veri kümeniz üzerinde bir model eğitmek istiyorsanız, önce resimler toplamalısınız. İdeal olarak, her sınıf için 100 resim kullanmak istersiniz. Örneğin, bir kedi ve köpek dedektörü eğitiyorsunuz diyelim. 100 kedi resmi ve 100 köpek resmi toplamanız gerekecek. Hap resimleri için, sadece internette arama yaptım ve çeşitli resimler indirdim. Ancak kendi veri kümeniz için, farklı arka planlara ve açılara sahip çeşitli resimler çekmenizi tavsiye ederim.
<p align="left">
  <img src="doc/1c84d1d5-2318-5f9b-e054-00144ff88e88.jpg">
</p>
<p align="left">
  <img src="doc/5mg-325mg_Hydrocodone-APAP_Tablet.jpg">
</p>
<p align="left">
  <img src="doc/648_pd1738885_1.jpg">
</p>

Bazı resimler topladıktan sonra, veri kümesini bölümlere ayırmalısınız. Bununla, verileri bir eğitim kümesine ve bir test kümesine ayırmanız gerektiğini kastediyorum. Resimlerinizin %80'ini ```images\training``` klasörüne ve kalan %20'sini ```images\test``` klasörüne koymalısınız. Resimlerinizi ayırdıktan sonra, [LabelImg](https://tzutalin.github.io/labelImg) ile etiketleyebilirsiniz.

LabelImg'yi indirdikten sonra, Open Dir ve Save Dir gibi ayarları yapılandırın. Bu, tüm resimler arasında gezinmenize ve nesnelerin etrafında sınırlayıcı kutular ve etiketler oluşturmanıza olanak tanır. Resminizi etiketledikten sonra kaydettiğinizden ve bir sonraki resme geçtiğinizden emin olun. Bunu ```images\test``` ve ```images\train``` klasörlerindeki tüm resimler için yapın.

<p align="left">
  <img src="doc/labelimg.png">
</p>

Artık veri kümemizi topladık. Bu, eğitim verilerini oluşturmaya hazır olduğumuz anlamına gelir. Öyleyse bir sonraki adıma geçelim!

### Eğitim Verilerinin Oluşturulması

Görüntülerimiz ve XML dosyalarımız hazır olduğundan, label_map'i oluşturmaya hazırız. Annotations klasöründe bulunur, bu yüzden Dosya Gezgini içinde ona gidin. label_map.pbtxt'yi bulduktan sonra, istediğiniz bir Metin Düzenleyicisi ile açın. Benim Hap Sınıflandırma Modelimi kullanmayı planlıyorsanız, herhangi bir değişiklik yapmanıza gerek yoktur ve boru hattını yapılandırmaya atlayabilirsiniz. Kendi özel nesne dedektörünüzü yapmak istiyorsanız, etiketlerinizin her biri için benzer bir öğe oluşturmanız gerekir. Modelimde iki sınıf hap olduğu için, labelmap'im şuna benziyordu:
```
item {
    id: 1
    name: 'Acetaminophen 325 MG Oral Tablet'
}

item {
    id: 2
    name: 'Ibuprofen 200 MG Oral Tablet'
}
```
Örneğin, bir basketbol, futbol ve beyzbol dedektörü yapmak isteseydiniz, labelmap'iniz şuna benzerdi:
```
item {
    id: 1
    name: 'basketball'
}

item {
    id: 2
    name: 'football'
}

item {
    id: 3
    name: 'baseball'
}
```
Bunu tamamladıktan sonra ```label_map.pbtxt``` olarak kaydedin ve metin düzenleyiciden çıkın. Şimdi eğitim için RECORD dosyaları oluşturmalıyız. Bunu yapma betiği C:\TensorFlow\scripts\preprocessing dizininde bulunur, ancak önce şu komutla pandas paketini yüklemeliyiz:

```
pip install pandas
```
Şimdi şu komutla scripts\preprocessing dizinine gitmeliyiz:

```
cd C:\TensorFlow\scripts\preprocessing
```

Doğru dizine girdikten sonra, kayıtları oluşturmak için şu iki komutu çalıştırın:

```
python generate_tfrecord.py -x C:\Tensorflow\workspace\training_demo\images\train -l C:\Tensorflow\workspace\training_demo\annotations\label_map.pbtxt -o C:\Tensorflow\workspace\training_demo\annotations\train.record

python generate_tfrecord.py -x C:\Tensorflow\workspace\training_demo\images\test -l C:\Tensorflow\workspace\training_demo\annotations\label_map.pbtxt -o C:\Tensorflow\workspace\training_demo\annotations\test.record
```
 Her komuttan sonra, TFRecord Dosyasının oluşturulduğunu belirten bir başarı mesajı almalısınız. Yani şimdi ```annotations``` altında bir ```test.record``` ve ```train.record``` olmalıdır. Bu, gerekli tüm verileri oluşturduğumuz anlamına gelir ve bir sonraki adımda eğitim hattını yapılandırmaya geçebiliriz.

### Eğitim Hattının Yapılandırılması
Bu eğitim için, TensorFlow önceden eğitilmiş modellerinden birinden bir CONFIG Dosyası kullanacağız. [TensorFlow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)'da çok sayıda model var, ancak biz spektrumun daha hızlı ucunda ve iyi bir performansa sahip olan [SSD MobileNet V2 FPNLite 640x640](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz)'ı kullanacağız. İsterseniz farklı bir model seçebilirsiniz, ancak adımları biraz değiştirmeniz gerekecek.

İstediğiniz modeli indirmek için TensorFlow Model Zoo'da adına tıklamanız yeterlidir. Bu, bir tar.gz dosyası indirmelidir. İndirildikten sonra, dosyanın içeriğini ```pre-trained-models``` dizinine çıkarın. Bu dizinin yapısı şimdi şöyle görünmelidir:

```
training_demo/
├─ ...
├─ pre-trained-models/
│  └─ ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/
│     ├─ checkpoint/
│     ├─ saved_model/
│     └─ pipeline.config
└─ ...
```
Şimdi, eğitim hattımızı saklamak için bir dizin oluşturmalıyız. ```models``` dizinine gidin ve ```my_ssd_mobilenet_v2_fpnlite``` adlı bir klasör oluşturun. Ardından, daha önce indirdiğimiz önceden eğitilmiş modelden ```pipeline.config``` dosyasını yeni oluşturduğumuz dizine kopyalayın. Dizin yapınız şimdi şöyle görünmelidir:

```
training_demo/
├─ ...
├─ models/
│  └─ my_ssd_mobilenet_v2_fpnlite/
│     └─ pipeline.config
└─ ...
```

Ardından, ```models\my_ssd_mobilenet_v2_fpnlite\pipeline.config``` dosyasını bir metin düzenleyicide açın çünkü bazı değişiklikler yapmamız gerekiyor.
- Satır 3. ```num_classes``` değerini modelinizin algıladığı sınıf sayısına değiştirin. Basketbol, beyzbol ve futbol örneği için ```num_classes: 3``` olarak değiştirirsiniz.
- Satır 135. ```batch_size``` değerini kullanılabilir belleğe göre değiştirin (Daha yüksek değerler daha fazla bellek gerektirir ve tam tersi). Ben şu şekilde değiştirdim:
  - ```batch_size: 6```
- Satır 165. ```fine_tune_checkpoint``` değerini şu şekilde değiştirin:
  - ```fine_tune_checkpoint: "pre-trained-models/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/checkpoint/ckpt-0"```
- Satır 171. ```fine_tune_checkpoint_type``` değerini şu şekilde değiştirin:
  - ```fine_tune_checkpoint_type: "detection"```
- Satır 175. ```label_map_path``` değerini şu şekilde değiştirin:
  - ```label_map_path: "annotations/label_map.pbtxt"```
- Satır 177. ```input_path``` değerini şu şekilde değiştirin:
  - ```input_path: "annotations/train.record"```
- Satır 185. ```label_map_path``` değerini şu şekilde değiştirin:
  - ```label_map_path: "annotations/label_map.pbtxt"```
- Satır 189. ```input_path``` değerini şu şekilde değiştirin:
  - ```input_path: "annotations/test.record"```

Gerekli tüm değişiklikleri yaptıktan sonra, eğitime hazırız demektir. O halde bir sonraki adıma geçelim!
### Modeli Eğitme
Şimdi Anaconda Komut İsteminize geri dönün. Şu komutla ```training_demo``` dizinine geçin:

```
cd C:\TensorFlow\workspace\training_demo
```

Eğitim betiğini zaten dizine taşıdım, bu yüzden çalıştırmak için şunu kullanmanız yeterlidir:

```
python model_main_tf2.py --model_dir=models\my_ssd_mobilenet_v2_fpnlite --pipeline_config_path=models\my_ssd_mobilenet_v2_fpnlite\pipeline.config
```

Betiği çalıştırırken, birkaç uyarı beklemeniz gerekir, ancak hata olmadıkları sürece bunları görmezden gelebilirsiniz. Sonunda eğitim süreci başladığında, şuna benzer bir çıktı görmelisiniz:

```
INFO:tensorflow:Step 100 per-step time 0.640s loss=0.454
I0810 11:56:12.520163 11172 model_lib_v2.py:644] Step 100 per-step time 0.640s loss=0.454
```

Tebrikler! Modelinizi resmi olarak eğitmeye başladınız! Şimdi arkanıza yaslanıp rahatlayabilirsiniz çünkü bu, sisteminize bağlı olarak birkaç saat sürecektir. Daha önce bahsettiğim özelliklerimle, eğitim yaklaşık 2 saat sürdü. TensorFlow, sürecin her 100 adımında yukarıdakine benzer bir çıktı kaydeder, bu nedenle donmuş gibi görünüyorsa endişelenmeyin. Bu çıktı size iki istatistik gösterir: adım başına süre ve kayıp. Kayba dikkat etmek isteyeceksiniz. Kayıtlar arasında kayıp azalma eğilimindedir. İdeal olarak, programı 0.150 ile 0.200 arasındayken durdurmak isteyeceksiniz. Bu, yetersiz uydurmayı ve aşırı uydurmayı önler. Benim için, kaybın bu aralığa girmesi yaklaşık 4000 adım sürdü. Ve sonra programı durdurmak için CTRL+C kullanmanız yeterlidir.

### TensorBoard ile Eğitimi İzleme (İsteğe Bağlı)

TensorFlow, TensorBoard ile eğitimi izlemenize ve eğitim metriklerini görselleştirmenize olanak tanır! Bunun tamamen isteğe bağlı olduğunu ve eğitim sürecini etkilemeyeceğini unutmayın, bu nedenle yapmak isteyip istemediğiniz size kalmıştır.
İlk olarak, yeni bir Anaconda Komut İstemi açın. Ardından, yapılandırdığımız sanal ortamı şu komutla etkinleştirin:

```
conda activate tensorflow
```

Ardından, şu komutla ```training_demo``` dizinine geçin:

```
cd C:\TensorFlow\workspace\training_demo
```
Bir TensorBoard Sunucusu başlatmak için şunu kullanın:

```
tensorboard --logdir=models\my_ssd_mobilenet_v2_fpnlite
```
Şuna benzer bir çıktı vermelidir:

```
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.2.2 at http://localhost:6006/ (Press CTRL+C to quit)
```

Ardından, bir web tarayıcısı açın ve verilen URL'yi arama çubuğuna yapıştırın. Bu sizi, eğitimi sürekli olarak izleyebileceğiniz TensorBoard Sunucusuna götürmelidir!

### Çıkarım Grafiğini Dışa Aktarma

Eğitimi bitirdikten ve betiği durdurduktan sonra, bitmiş modelinizi dışa aktarmaya hazırsınız demektir! Hala ```training_demo``` dizininde olmalısınız, ancak değilseniz şu komutla geçin:

```
cd C:\TensorFlow\workspace\training_demo
```

Dışa aktarmak için gereken betiği zaten taşıdım, bu yüzden tek yapmanız gereken şu komutu çalıştırmak:

```
python .\exporter_main_v2.py --input_type image_tensor --pipeline_config_path .\models\my_ssd_mobilenet_v2_fpnlite\pipeline.config --trained_checkpoint_dir .\models\my_ssd_mobilenet_v2_fpnlite\ --output_directory .\exported-models\my_mobilenet_model
```

**```TypeError: Expected Operation, Variable, or Tensor, got block4 in exporter_main_v2.py``` benzeri bir hata alırsanız, [bu](https://github.com/tensorflow/models/issues/8881) hata konusuna bakın**

Ancak bu program başarıyla tamamlanırsa, tebrikler çünkü modeliniz bitti! ```C:\TensorFlow\workspace\training_demo\exported-models\my_mobilenet_model\saved_model``` klasöründe bulunmalıdır. ```saved_model.pb``` adlı bir PB Dosyası olmalıdır. Bu, çıkarım grafiğidir! Ayrıca ```label_map.pbtxt``` dosyasını bu dizine kopyalamayı tercih ediyorum çünkü test için işleri biraz daha kolaylaştırıyor. Labelmap'in nerede olduğunu unuttuysanız, ```C:\TensorFlow\workspace\training_demo\annotations\label_map.pbtxt``` içinde olmalıdır. Labelmap ve çıkarım grafiği düzenlendiğine göre, test etmeye hazırız!

### Modeli Değerlendirme (İsteğe Bağlı)

IoU, mAP, Geri Çağırma ve Hassasiyet gibi model metriklerini ölçmek istiyorsanız, bu adımı tamamlamak isteyeceksiniz. Modeli değerlendirmek için en güncel TensorFlow Belgeleri [burada](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_training_and_evaluation.md#evaluation) bulunacaktır.

Hala ```TensorFlow/workspace/training_demo``` dizininde olmalısınız, ancak değilseniz şu komutla geçin:

```
cd C:\TensorFlow\workspace\training_demo
```

Şimdi değerlendirme için aşağıdaki komutu çalıştırmanız yeterlidir:

```
python model_main_tf2.py --pipeline_config_path models\my_ssd_mobilenet_v2_fpnlite\pipeline.config --model_dir models\my_ssd_mobilenet_v2_fpnlite --checkpoint_dir models\my_ssd_mobilenet_v2_fpnlite --alsologtostderr
```

**```TypeError: object of type <class 'numpy.float64'> cannot be safely interpreted as an integer``` benzeri bir hata alırsanız, NumPy sürümünüzü düşürmeniz yeterlidir. Benim için 1.17.3 sürümü çalıştı, bu yüzden ```pip install numpy==1.17.3``` ile yükleyebilirsiniz**

Her şey düzgün çalışıyorsa, şuna benzer bir şey almalısınız:

<p align="center">
  <img src="doc/evaluation.png">
</p>

### Bitmiş Modeli Test Etme

Modelinizi test etmek için, ```TF-image-od.py``` adlı sağladığım örnek nesne algılama betiğini kullanabilirsiniz. Bu, ```C:\TensorFlow\workspace\training_demo``` içinde bulunmalıdır. **Güncelleme**: Video desteği, argüman desteği ve fazladan bir OpenCV yöntemi ekledim. Her programın açıklaması aşağıda listelenecektir:
- ```TF-image-od.py```: Bu program, etiketleri ve sınırlayıcı kutuları görselleştirmek için viz_utils modülünü kullanır. Tek bir görüntü üzerinde nesne algılama gerçekleştirir ve bunu bir cv2 penceresiyle görüntüler.
- ```TF-image-object-counting.py```: Bu program ayrıca tek bir görüntü üzerinde çıkarım gerçekleştirir. Tercih ettiğim OpenCV ile kendi etiketleme yöntemimi ekledim. Ayrıca algılama sayısını sayar ve sol üst köşede görüntüler. Son görüntü, yine bir cv2 penceresiyle görüntülenir.
- ```TF-video-od.py```: Bu program ```TF-image-od.py```'ye benzer. Ancak, bir videonun her bir karesi üzerinde çıkarım gerçekleştirir ve bunu cv2 penceresi aracılığıyla görüntüler.
- ```TF-video-object-counting.py```: Bu program ```TF-image-object-counting.py```'ye benzer ve OpenCV ile benzer bir etiketleme yöntemine sahiptir. Girdi olarak bir video alır ve ayrıca her kare üzerinde nesne algılama gerçekleştirir ve algılama sayısını sol üst köşede görüntüler.

Her programın kullanımı şöyle görünür:

```
usage: TF-image-od.py [-h] [--model MODEL] [--labels LABELS] [--image IMAGE] [--threshold THRESHOLD]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Kaydedilmiş Modelin Bulunduğu Klasör
  --labels LABELS       Labelmap'in Bulunduğu Yer
  --image IMAGE         Algılama gerçekleştirmek için tek bir görüntünün adı
  --threshold THRESHOLD Algılanan nesneleri görüntülemek için minimum güven eşiği
```
Model veya labelmap, benim koyduğum yerlerden başka bir yerde bulunuyorsa, bu argümanlarla konumu belirtebilirsiniz. Ayrıca üzerinde çıkarım yapmak için bir görüntü/video sağlamalısınız. Benim Hap Algılama Modelimi kullanıyorsanız, varsayılan değer iyi olacağından bu gereksizdir. Video betiklerinden birini kullanıyorsanız, ```--image``` yerine ```--video``` kullanın ve test videonuzun yolunu sağlayın. Örneğin, aşağıdaki adımlar örnek ```TF-image-od.py``` betiğini çalıştırır.

```
cd C:\TensorFlow\workspace\training_demo
```

Ardından betiği çalıştırmak için şunu kullanmanız yeterlidir:

```
python TF-image-od.py
```

**Şuna benzer bir hata alırsanız: ```
cv2.error: OpenCV(4.3.0) C:\Users\appveyor\AppData\Local\Temp\1\pip-req-build-kv3taq41\opencv\modules\highgui\src\window.cpp:651: error: (-2:Belirtilmemiş hata) İşlev uygulanmadı. Kütüphaneyi Windows, GTK+ 2.x veya Cocoa desteğiyle yeniden oluşturun. Ubuntu veya Debian kullanıyorsanız, libgtk2.0-dev ve pkg-config'i kurun, ardından cmake veya yapılandırma betiğini 'cvShowImage' işlevinde yeniden çalıştırın
``` bu durumda ```pip install opencv-python``` komutunu çalıştırın ve programı tekrar çalıştırın**

Her şey düzgün çalışıyorsa, şuna benzer bir çıktı almalısınız:
<p align="center">
  <img src="doc/output.png">
</p>

Bu, işimizin bittiği anlamına gelir! Önümüzdeki birkaç hafta veya ay boyunca, yeni programlar üzerinde çalışmaya ve test etmeye devam edeceğim! Harika bir şey bulursanız, başkaları da öğrenebileceği için paylaşmaktan çekinmeyin! Herhangi bir hatanız varsa, bir sorun oluşturun, memnuniyetle incelerim. Tebrikler ve bir dahaki sefere kadar hoşça kalın!

