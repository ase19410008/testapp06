""" Streamlitによる退職予測AIシステムの開発
"""

from itertools import chain
import numpy as np
import pandas as pd 
import streamlit as st
import matplotlib.pyplot as plt 
import japanize_matplotlib
import seaborn as sns 

# 決定木
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# ランダムフォレスト
from sklearn.ensemble import RandomForestClassifier

# 精度評価用
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

# データを分割するライブラリを読み込む
from sklearn.model_selection import train_test_split

# データを水増しするライブラリを読み込む
from imblearn.over_sampling import SMOTE

# ロゴの表示用
from PIL import Image

# ディープコピー
import copy

sns.set()
japanize_matplotlib.japanize()  # 日本語フォントの設定

# matplotlib / seaborn の日本語の文字化けを直す、汎用的かつ一番簡単な設定方法 | BOUL
# https://boul.tech/mplsns-ja/


def st_display_table(df: pd.DataFrame):
    """
    Streamlitでデータフレームを表示する関数
    
    Parameters
    ----------
    df : pd.DataFrame
        対象のデータフレーム

    Returns
    -------
    なし
    """

    # データフレームを表示
    st.subheader('データの確認')
    st.table(df)

    # 参考：Streamlitでdataframeを表示させる | ITブログ
    # https://kajiblo.com/streamlit-dataframe/


def st_display_graph(df: pd.DataFrame, x_col : str):
    """
    Streamlitでグラフ（ヒストグラム）を表示する関数
    
    Parameters
    ----------
    df : pd.DataFrame
        対象のデータフレーム
    x_col : str
        対象の列名（グラフのx軸）

    Returns
    -------
    なし
    """

    # グラフ（ヒストグラム）の設定
    # sns.countplot(data=df, x=x_col, ax=ax)
    # sns.countplot(data=df, x=x_col)

    fig, ax = plt.subplots()    # グラフの描画領域を準備
    plt.grid(True)              # 目盛線を表示する

    ax.hist(df[x_col])
    st.pyplot(fig)              # Streamlitでグラフを表示する


def ml_dtree(
    X: pd.DataFrame,
    y: pd.Series,
    depth: int) -> list:
    """
    決定木で学習と予測を行う関数
    
    Parameters
    ----------
    X : pd.DataFrame
        説明変数の列群
    y : pd.Series
        目的変数の列
    depth : int
        決定木の深さ

    Returns
    -------
    list: [学習済みモデル, 予測値, 正解率]
    """

    # 決定木モデルの生成（オプション:木の深さ）
    clf = DecisionTreeClassifier(max_depth=depth)

    # 学習
    clf.fit(X, y)

    # 予測
    pred = clf.predict(X)

    # accuracyで精度評価
    score = round(accuracy_score(y, pred), 2)
    recall = round(recall_score(y, pred, pos_label="Yes"), 2)
    f1 = round(f1_score(y, pred, pos_label="Yes"), 2)

    return [clf, pred, score, recall, f1]

def ml_rf(X, Y):
    rf = RandomForestClassifier(random_state=0)

    rf.fit(X, Y)

    rf_pred = rf.predict(X)

    score = round(accuracy_score(Y, rf_pred), 1)
    recall = round(recall_score(Y, rf_pred, pos_label="Yes"), 2)
    f1 = round(f1_score(Y, rf_pred, pos_label="Yes"), 2)

    return [rf, rf_pred, score, recall, f1]

def st_display_dtree(clf, features):
    """
    Streamlitで決定木のツリーを可視化する関数
    
    Parameters
    ----------
    clf : 
        学習済みモデル
    features :
        説明変数の列群

    Returns
    -------
    なし
    """

    # 必要なライブラリのインポート    
    from sklearn.tree import plot_tree

    # 可視化する決定木の生成
    plot_tree(clf, feature_names=features, class_names=True, filled=True)

    # Streamlitで決定木を表示する
    st.pyplot(plt)

    # # 可視化する決定木の生成
    # dot = tree.export_graphviz(clf, 
    #     # out_file=None,  # ファイルは介さずにGraphvizにdot言語データを渡すのでNone
    #     # filled=True,    # Trueにすると、分岐の際にどちらのノードに多く分類されたのか色で示してくれる
    #     # rounded=True,   # Trueにすると、ノードの角を丸く描画する。
    # #    feature_names=['あ', 'い', 'う', 'え'], # これを指定しないとチャート上で特徴量の名前が表示されない
    #     # feature_names=features, # これを指定しないとチャート上で説明変数の名前が表示されない
    # #    class_names=['setosa' 'versicolor' 'virginica'], # これを指定しないとチャート上で分類名が表示されない
    #     # special_characters=True # 特殊文字を扱えるようにする
    #     )

    # # Streamlitで決定木を表示する
    # st.graphviz_chart(dot)

def st_display_rf(rf, train_x):
    feat_importances = pd.Series(rf.feature_importances_, index=train_x).sort_values()
    feat_importances = feat_importances.to_frame(name="重要度").sort_values(by='重要度', ascending=False)

    feat_importances[0:20].sort_values(by="重要度").plot.barh()
    plt.legend(loc="lower right")
    st.pyplot(plt)


def main():
    """ メインモジュール
    """

    # stのタイトル表示
    st.title("退職予測AI\n（Machine Learning)")

    # サイドメニューの設定
    activities = ["データ確認", "要約統計量", "グラフ表示", "学習と検証", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    # 未アップロード
    if 'df' not in st.session_state:
        # ファイルのアップローダー
        uploaded_file = st.sidebar.file_uploader("訓練用データのアップロード", type='csv') 

        # アップロードの有無を確認
        if uploaded_file is not None:

            # 一度、read_csvをするとインスタンスが消えるので、コピーしておく
            ufile = copy.deepcopy(uploaded_file)

            try:
                # 文字列の判定
                pd.read_csv(ufile, encoding="utf_8_sig")
                enc = "utf_8_sig"
            except:
                enc = "shift-jis"

            finally:
                # データフレームの読み込み
                df = pd.read_csv(uploaded_file, encoding=enc) 

                # データフレームをセッションステートに退避（名称:df）
                st.session_state.df = copy.deepcopy(df)
        else:
            st.subheader('訓練用データをアップロードしてください')

    if choice == activities[0]:        
        # セッションステートにデータフレームがあるかを確認
        if 'df' in st.session_state:

            # セッションステートに退避していたデータフレームを復元
            df = copy.deepcopy(st.session_state.df)

            # スライダーの表示（表示件数）
            cnt = st.sidebar.slider('表示する件数', 1, len(df), 10)

            # テーブルの表示
            st_display_table(df.head(int(cnt)))

    if choice == activities[1]:

        # セッションステートにデータフレームがあるかを確認
        if 'df' in st.session_state:

            # セッションステートに退避していたデータフレームを復元
            df = copy.deepcopy(st.session_state.df)

            # 要約統計量の表示
            st.table(df.describe())

    if choice == activities[2]:

        # セッションステートにデータフレームがあるかを確認
        if 'df' in st.session_state:

            # セッションステートに退避していたデータフレームを復元
            df = copy.deepcopy(st.session_state.df)

            x_col = st.sidebar.selectbox('グラフのx軸', df.columns)
            # グラフの表示
            st_display_graph(df=df, x_col=x_col)

    if choice == activities[3]:

        if 'df' in st.session_state:

            # セッションステートに退避していたデータフレームを復元
            df = copy.deepcopy(st.session_state.df)

            train_X, valid_X, train_Y, valid_Y = train_test_split(df.drop("退職", axis=1), df["退職"], test_size=.3, random_state=0)

            oversample = SMOTE(sampling_strategy=.5, random_state=0)
            train_X_over, train_Y_over = oversample.fit_resample(train_X, train_Y)

            # 説明変数と目的変数の設定
            # train_X = df.drop("退職", axis=1)   # 退職列以外を説明変数にセット
            # train_Y = df["退職"]                # 退職列を目的変数にセット

            ml_type = ['決定木', 'ランダムフォレスト']
            clf = st.sidebar.selectbox('学習の手法', ml_type)

            if clf == ml_type[0]:
                depth = st.sidebar.selectbox('決定木の深さ(サーバーの負荷軽減の為Max=3)', range(1, 4))
                # 決定木による予測
                clf, train_pred, train_scores, recall, f1 = ml_dtree(train_X_over, train_Y_over, depth)

                clf, valid_pred, valid_scores, rec2, f2 = ml_dtree(valid_X, valid_Y, depth)

                st.caption("決定木の可視化")
                # 決定木のツリーを出力
                st_display_dtree(clf, train_X.columns)
            else:
                rf, train_pred, train_scores, recall, f1 = ml_rf(train_X_over, train_Y_over)

                rf, valid_pred, valid_scores, rec2, f2 = ml_rf(valid_X, valid_Y)

                st.caption("重要度の可視化")
                st_display_rf(rf, train_X.columns)

            st.subheader("検証用データでの予測精度")
            # 正解率を出力
            st.caption("AIの予測が「全員、退職しない」に偏った場合は(意味がないので)全ての精度は0で表示されます")
            train_col1, train_col2, train_col3 = st.columns(3)
            train_col1.metric('正解率', train_scores)
            train_col2.metric('再現率', recall)
            train_col3.metric('適合率', f1)

            st.subheader("訓練用データでの予測精度")
            test_col1, test_col2, test_col3 = st.columns(3)
            test_col1.metric('正解率', valid_scores)
            test_col2.metric('再現率', rec2)
            test_col3.metric('適合率', f2)

    elif choice == activities[4]:
        st.image("logo.png")
        'recall_scoreで参考にしたもの'
        "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html"
        'https://stackoverflow.com/questions/35953300/valueerror-data-is-not-binary-and-pos-label-is-not-specified-for-roc-curve'

if __name__ == "__main__":
    main()