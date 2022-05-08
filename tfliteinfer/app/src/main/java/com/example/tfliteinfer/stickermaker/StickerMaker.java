package com.example.tfliteinfer.stickermaker;

import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.net.Uri;
import android.os.Environment;
import android.util.Log;
import android.widget.Toast;

import com.example.tfliteinfer.R;

import org.tensorflow.lite.support.model.Model;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;

public class StickerMaker {
    public void init(float [] output) {
    }
//    private void heapter_glass() {
//
//    }
    public void make_sticker(Canvas tempCanvas, Bitmap sticker, float[] output, Paint paint) {
        float left_x = output[14];
        float left_y = output[15];
        float right_x = output[6];
        float right_y = output[7];

//                float dist = (float) (right_x - left_x);    // naive distance
        float dist = (float) Math.sqrt((right_x-left_x)*(right_x-left_x)+(right_y-left_y)*(right_y-left_y));     // euclidean distance
        int width = (int) (dist*2.2);
        int height = (int) (dist*0.6);
//                int start_x = (int) (left_x+right_x)/2 - width/2;
//                int start_y = (int) (left_y+right_y)/2 - height/2;
        int start_x = (int) ((left_x+right_x)/2 - width/2);
        int start_y = (int) ((left_y+right_y)/2 - height/2);

//        paint.setColor(Color.RED);
//        tempCanvas.drawCircle(start_x, start_y, 8, paint);
//        tempCanvas.drawCircle(0, 0, 8, paint);
//        tempCanvas.drawCircle(left_x,left_y,8,paint);
//        tempCanvas.drawCircle(right_x,right_y,8,paint);
//        tempCanvas.drawCircle((left_x+right_x)/2,(left_y+right_y)/2, 8, paint);

        Matrix matrix = new Matrix();
        float rotation = (float) Math.atan2(right_y-left_y,right_x-left_x) * 180/(float)Math.PI; // rotation degree
        matrix.postRotate(rotation,(left_x+right_x)/2, (left_y+right_y)/2);

        Bitmap sticker2 = Bitmap.createScaledBitmap(sticker,width, height, false); //이미지 리사이징 실행코드

        Bitmap rotatedGlasses = Bitmap.createBitmap(sticker2, 0, 0, sticker2.getWidth(), sticker2.getHeight(), matrix, true);
        int w = Math.abs(rotatedGlasses.getWidth() - width);
        int h = Math.abs(rotatedGlasses.getHeight() - height);

        start_x -= (int) w / 2;
        start_y -= (int) h / 2;

//                Bitmap glasses3 = glasses2.copy(Bitmap.Config.ARGB_8888, false);
//                rotation
        tempCanvas.drawBitmap(rotatedGlasses, (int) start_x, (int) start_y, null);
//        return tempCanvas;
    }

    public void saveBitmaptoJpeg(Bitmap bitmap, Context context) {
//        Log.d(TAG, "dddddddddsaveBitmaptoJpeg: "+name);
        SimpleDateFormat simpleDateFormat = new SimpleDateFormat("yyyyMMdd-hhmmss");
        FileOutputStream out = null;
        String saveDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM).toString()+ "/azzit";
        File uploadFolder = new File(saveDir);
//        File uploadFolder = Environment.getExternalStoragePublicDirectory("/DCIM/azzit");
        if (!uploadFolder.exists()) { //만약 경로에 폴더가 없다면
            uploadFolder.mkdirs(); //폴더 생성
        }
        String filename = "azzit_"+ simpleDateFormat.format(new Date()) + ".png";
        File temp_file = new File(saveDir, filename);
        try {
            out = new FileOutputStream(temp_file);
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, out); // bmp is your Bitmap instance
            Toast.makeText(context.getApplicationContext(), "사진이 저장되었어요.", Toast.LENGTH_SHORT).show();
            Intent mediaScanIntent = new Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE);
            Uri contentUri = Uri.fromFile(temp_file);
            mediaScanIntent.setData(contentUri);
            context.sendBroadcast(mediaScanIntent);
        } catch (FileNotFoundException e) {
            //e.printStackTrace()
//                renameFile(bitmap, Str_Path, title);
            Toast.makeText(context.getApplicationContext(), "사진이 저장되지 않았어요.", Toast.LENGTH_SHORT).show();
        } catch (Exception e) {
            e.printStackTrace();
            Toast.makeText(context.getApplicationContext(), "사진이 저장되지 않았어요.", Toast.LENGTH_SHORT).show();
        } finally {
            try {
                if (out != null) {
                    out.close();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

    }
}
