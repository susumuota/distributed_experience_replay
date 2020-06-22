# 分散計算フレームワークRayによる分散型強化学習実装の試み Distributed Experience Replay with Ray

分散型経験リプレイ (distributed experience replay) を使ったシンプルな分散型強化学習の Ray による実装です。

Ape-X[1] の Actor / Learner アーキテクチャによる経験の生成と学習の分離と、ε_i グリーディを実装しています(それ以外は実装していません)。


このソースコード/設定ファイルは以下の勉強会のために準備したものです。

https://do2dle.connpass.com/event/178184/

また、勉強会当日の発表資料は以下にあります。

[![thumbnail](https://image.slidesharecdn.com/distributedrlota20200622-200622061616/95/slide-1-1024.jpg)](https://www.slideshare.net/SusumuOTA/distributed-rl-ota-20200622)


## インストール

テスト環境は macOS 10.15.5 + Python 3.7.7 です。

```
git clone https://github.com/susumuota/distributed_experience_replay.git
cd distributed_experience_replay
python3 -m venv venv
source venv/bin/activate
python -V  # Python 3.7.7
pip install -r requirements.txt
```

requirements.txt の tensorflow と keras のバージョン指定を外すと tensorflow 2 環境で実行可能ですが、なるべく Amazon EC2 で使うイメージ(Deep Learning AMI (Ubuntu))に入っている tensorflow や keras 等のバージョンに合わせておいたほうが良いです。

## アンインストール

```
deactivate
rm -rf venv
```


## Ray による分散型経験リプレイを使った強化学習の実装

学習する環境は CartPole-v0 です。

### ローカルマシンでの実行方法

Actor の数を main() 内の変数 `n` で指定できますので、各自の実行環境のCPUコア数に合わせて変更すると良いです。

```
python distributed_experience_replay.py
```

## Ray による Amazon EC2 のクラスタ構築

AWS でのクラスターの構築方法ドキュメントを参考に進めます。

https://docs.ray.io/en/master/autoscaling.html#aws

**以下は、IAM でユーザを追加/削除/設定したり、EC2 で手動でサーバを開始/終了/設定出来る方を想定しています。**

**以下は、あくまでも私が確認できた範囲ですので、不十分あるいは問題がある可能性があります。各自で Ray のドキュメントやソースコードを確認して、各自の責任において実行してください。**

### AWS 側の設定

- IAMコンソールでユーザ追加
  - 「アクセスの種類」を「プログラムによるアクセス」→アクセスキーIDとシークレットアクセスキーが生成
  - 「既存のポリシーを直接アタッチ」で「**AmazonEC2FullAccess**」と「**IAMFullAccess**」を追加
- アクセスキーIDとシークレットアクセスキーが作成できたらローカルマシンの ~/.aws/credentials に保存
- ポリシーの部分は **要検討**
  - Rayのドキュメントにはどのポリシーが必要か具体的には書いてない(or 私が見つけられなかった)ので、各自で適切なポリシーを設定してください
  - 私の場合は、実験直前にユーザを作成して実験が終わったらすぐユーザを削除しています



### ローカルマシン側の設定

今回実験に使用した設定ファイルはこちらです。

https://github.com/susumuota/distributed_experience_replay/blob/master/example-full.yaml

元ファイルは以下にあります。

https://github.com/ray-project/ray/blob/master/python/ray/autoscaler/aws/example-full.yaml

元ファイルとの diff は以下の通りです。

```
@@ -12,7 +12,7 @@
 # The initial number of worker nodes to launch in addition to the head
 # node. When the cluster is first brought up (or when it is refreshed with a
 # subsequent `ray up`) this number of nodes will be started.
-initial_workers: 0
+initial_workers: 2
 
 # Whether or not to autoscale aggressively. If this is enabled, if at any point
 #   we would start more workers, we start at least enough to bring us to
@@ -56,6 +56,7 @@
     # Nodes are currently spread between zones by a round-robin approach,
     # however this implementation detail should not be relied upon.
     availability_zone: us-west-2a,us-west-2b
+    # cache_stopped_nodes: False
 
 # How Ray will authenticate with newly launched nodes.
 auth:
@@ -112,6 +113,7 @@
 
 # List of shell commands to run to set up nodes.
 setup_commands:
+    - source activate tensorflow_p36
     # Note: if you're developing Ray, you probably want to create an AMI that
     # has your Ray repo pre-cloned. Then, you can replace the pip installs
     # below with a git checkout <your_sha> (and possibly a recompile).
@@ -123,10 +125,12 @@
     # - sudo pkill -9 apt-get || true
     # - sudo pkill -9 dpkg || true
     # - sudo dpkg --configure -a
+    - pip install boto3
+    - pip install gym
+    - pip install ray[rllib]
 
 # Custom commands that will be run on the head node after common setup.
-head_setup_commands:
-    - pip install boto3==1.4.8  # 1.4.8 adds InstanceMarketOptions
+head_setup_commands: []
 
 # Custom commands that will be run on worker nodes after common setup.
 worker_setup_commands: []
```

### クラスタの立ち上げ

```
ray up example-full.yaml
```

### クラスタでスクリプト実行

クラスタで実行する場合は distributed_experience_replay.py の ray.init の引数に `address='auto'` を追加してください。

```
ray.init(address='auto') # for cluster
# ray.init() # for local machine
```

以下のコマンドで distributed_experience_replay.py スクリプトをクラスタで実行します。

```
ray submit example-full.yaml distributed_experience_replay.py
```

### クラスタの終了

```
ray down example-full.yaml
```


### AWSコンソールでクラスタの様子を確認

https://console.aws.amazon.com/

* ray up/submit した後にインスタンスを確認
* ray down した後にこの画面からインスタンスの状態を**終了(terminated)**にする
* 停止(stopped)だと**課金が継続するので注意**


### 注意点

**クラスタ自動構築時に作成されるもの(私が確認できた範囲)を削除**

- EC2
  - **インスタンス**
    - デフォルトyaml設定では ray downしてもインスタンスが「停止」されるだけなので実験が終わったら手動で「終了」する
    - yaml 設定で 「cache_stopped_nodes: False」を指定すると ray down時に「停止」ではなく「終了」するようになる
  - **ボリューム**
    - インスタンス開始とともに作成され、終了で削除されるはず
  - キーペア
    - ray-autoscaler*
  - セキュリティグループ
    - ray-autoscaler*
  - IAM
    - ロール
      - ray-autoscaler-v1
  - ssh 秘密鍵
    - ~/.ssh/ray-autoscaler*.pem

**IAMコンソールで自分で作ったユーザも削除**

**インスタンスとボリュームに関しては放っておくと課金が継続するのでAWSコンソールで最終確認**

**上記は、あくまでも私が確認できた範囲ですので、これ以外にも作成されるものがある可能性があります。各自で Ray のドキュメントやソースコードを確認して、各自で責任において対応してください。**


## RLlib の Ape-X 実行方法

以下のページを参考に rllib コマンドで yaml ファイル指定して実行します。

https://github.com/ray-project/rl-experiments

例: Atari を Ape-X で学習させるコマンド

```
rllib train -f atari-apex/atari-apex.yaml
```

また、rllib コマンドと yaml を使わずに Python スクリプトで実行するサンプルは以下の通りです。

```
python rllib_test.py
```


## 参考文献

[1] Horgan et al., "Distributed Prioritized Experience Replay", ICLR2018, 2018. https://openreview.net/forum?id=H1Dy---0Z

[2] The Ray Team, "Ray - Fast and Simple Distributed Computing", 2020.
https://ray.io/

[3] Moritz et al., "Ray: A Distributed Framework for Emerging AI Applications", 13th USENIX Symposium on Operating Systems Design and Implementation, 2018.
https://www.usenix.org/conference/osdi18/presentation/moritz

[4] Liang et al., "RLlib: Abstractions for Distributed Reinforcement Learning", ICML 2018, 2018.
https://arxiv.org/abs/1712.09381


## 免責

このドキュメント/リポジトリに記述されている情報は、各自の責任おいて利用/実行してください。


## Author

Susumu OTA
