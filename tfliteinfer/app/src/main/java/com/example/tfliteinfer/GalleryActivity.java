package com.example.tfliteinfer;

import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageDecoder;
import android.graphics.Paint;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.tfliteinfer.tflite.ClassifierWithModel;

import java.io.IOException;
import java.util.Arrays;

public class GalleryActivity extends AppCompatActivity {
    public static final String TAG = "[IC]GalleryActivity";
    public static final int GALLERY_IMAGE_REQUEST_CODE = 1;

    private ClassifierWithModel cls;
    private ImageView imageView;
    private TextView textView;
    private ImageView backimageView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_gallery);

        Button selectBtn = findViewById(R.id.selectBtn);
        selectBtn.setOnClickListener(v -> getImageFromGallery());

        imageView = findViewById(R.id.imageView);
        textView = findViewById(R.id.textView);

        cls = new ClassifierWithModel(this);
        try {
            cls.init();
        } catch (IOException ioe) {
            ioe.printStackTrace();
        }
    }

    private void getImageFromGallery(){
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT).setType("image/*");
//        Intent intent = new Intent(Intent.ACTION_PICK,
//                MediaStore.Images.Media.INTERNAL_CONTENT_URI);
        startActivityForResult(intent, GALLERY_IMAGE_REQUEST_CODE);
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {

        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == Activity.RESULT_OK &&
                requestCode == GALLERY_IMAGE_REQUEST_CODE) {
            if (data == null) {
                return;
            }

            Uri selectedImage = data.getData();
            Bitmap bitmap = null;

            try {
                if(Build.VERSION.SDK_INT >= 29) {
                    ImageDecoder.Source src =
                            ImageDecoder.createSource(getContentResolver(), selectedImage);
                    bitmap = ImageDecoder.decodeBitmap(src);
                } else {
                    bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), selectedImage);
                }
            } catch (IOException ioe) {
                Log.e(TAG, "Failed to read Image", ioe);
            }

            if(bitmap != null) {
                float[] output = cls.classify(bitmap); //모델 추론코드

//                내가 수정한 부분
                Bitmap bitmap_canvas = Bitmap.createBitmap(imageView.getWidth(), imageView.getHeight(), Bitmap.Config.ARGB_8888);
                Canvas tempCanvas = new Canvas(bitmap_canvas); //그림 넣을 캔버스 만들기

                int imageSize = 256;  // imageSize to rescale landmark

                //입력 이미지의 사이즈가 크기 때문에 이미지 뷰 영역에 맞춰줌
                float newWidth = bitmap.getWidth();
                float newHeight = bitmap.getHeight();
                if (bitmap.getWidth() >= imageView.getWidth()) {
                    newWidth = imageView.getWidth();
                    float tempWidth = bitmap.getWidth();
                    float tempHeight = bitmap.getHeight();
                    newHeight = ((float)(newWidth / tempWidth))*tempHeight;
                    Log.d(TAG, "onActivityResult: "+ newHeight);
                    if (newHeight >= imageView.getHeight()) {
                        tempHeight = newHeight;
                        newHeight = imageView.getHeight();
                        newWidth = (newHeight/tempHeight)*newWidth;
                    }
                }
                Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap,(int) newWidth, (int) newHeight, false); //이미지 리사이징 실행코드
                Bitmap targetBmp = resizedBitmap.copy(Bitmap.Config.ARGB_8888, false); //위 비트맵 이미지를 그냥 넣으면 오류떠서 오류 해결코드
                tempCanvas.drawBitmap(targetBmp, 0, 0, null); //캔버스에 입력 이미지를 넣음
                Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG); //그림을 그릴 페인트 생성
                paint.setColor(Color.CYAN); //점의 색 설정
                for (int index = 0; index <= 15;) { //점 찍는 반복문
                    tempCanvas.drawCircle(output[index]*newWidth/imageSize, output[index+1]*newHeight/imageSize, 8, paint);
                    output[index] = output[index]*newWidth/imageSize;
                    output[index+1] = output[index+1]*newHeight/imageSize;
                    index = index + 2;
                }


////////////////////////Sticker processing////////////////////////////////////

                float left_x = output[14];
                float left_y = output[15];
                float right_x = output[6];
                float right_y = output[7];

                float dist = (float) (right_x - left_x);    // 두 눈 사이 거리 euclidean 으로
                int width = (int) (dist*2.2);
                int height = (int) (dist*0.6);

//                int start_x = (int) (left_x+right_x)/2 - width/2;
//                int start_y = (int) (left_y+right_y)/2 - height/2;

                int start_x = (int) ((left_x+right_x)/2 - width/2);
                int start_y = (int) ((left_y+right_y)/2 - height/2);


                paint.setColor(Color.RED);
                tempCanvas.drawCircle(start_x, start_y, 8, paint);
                tempCanvas.drawCircle(0, 0, 8, paint);
                tempCanvas.drawCircle(left_x,left_y,8,paint);
                tempCanvas.drawCircle(right_x,right_y,8,paint);



                Bitmap glasses = BitmapFactory.decodeResource(getApplicationContext().getResources(), R.drawable.bitsunglass);
                Bitmap glasses2 = Bitmap.createScaledBitmap(glasses,width, height, false); //이미지 리사이징 실행코드
//                Bitmap glasses3 = glasses2.copy(Bitmap.Config.ARGB_8888, false);
//                rotation
                tempCanvas.drawBitmap(glasses2, (int) start_x, (int) start_y, null);

////////////////////////////////////////////////////////////




                imageView.setImageDrawable(new BitmapDrawable(getResources(), bitmap_canvas)); //입력이미지와 점을 이미지 뷰에 그려줌
                textView.setText(Arrays.toString(output)); //모델 추론 결과값 확인을 위한 텍스트 출력
//                끝
            }
        }
    }

    @Override
    protected void onDestroy() {
        cls.finish();
        super.onDestroy();
    }
}