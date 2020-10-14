from moviepy.editor import *

#img = ['1.jpg', '2.jpg']

#clips = [ImageClip(m).set_duration(2)
#      for m in img]

#concat_clip = concatenate_videoclips(clips, method="compose")
#concat_clip.write_videofile("test.mp4", fps=24)
import imageio

reader = imageio.get_reader('./datavideo/video.mp4')
fps = reader.get_meta_data()['fps']

writer = imageio.get_writer('./datavideo/video1.mp4', fps=fps)

for im in reader:
    print(im)
    writer.append_data(im)
writer.close()


############
'''
                cap = cv2.VideoCapture('./datavideo/video.mp4')

                # Define the codec and create VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'FMP4')
                # out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
                out = cv2.VideoWriter('./data/output.mp4', -1, 20.0, (720, 1280))
                i = 0
                while (cap.isOpened()):
                    ret, frame = cap.read()

                    if ret == True:
                        i = i + 1
                        print(type(frame))
                        cv2.imwrite(r"C:\Users\PRATIKA\Desktop\face-parsing.PyTorch\frames\frame"+str(i)+".jpg", frame)
                        frame = evaluate(dspth='./frames/frame'+str(i)+'.jpg', cp='79999_iter.pth')
                        cv2.imwrite(r"C:\Users\PRATIKA\Desktop\face-parsing.PyTorch\frames\frame"+str(i)+".jpg", frame)
                        #frame.show()
                        #frame = cv2.flip(frame, 0)
                        #print(frame.shape)
                        # write the flipped frame
                        out.write(frame)

                        #cv2.imshow('frame', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    else:
                        break

                # Release everything if job is finished
                cap.release()
                out.release()
                cv2.destroyAllWindows()
                '''

    import os
    import glob
    from natsort import natsorted
    from moviepy.editor import *

    base_dir = os.path.realpath("./frames")
    print(base_dir)

    gif_name = 'pic'
    fps = 40

    file_list = glob.glob('./frames/' + '*.jpg')  # Get all the pngs in the current directory
    file_list_sorted = natsorted(file_list, reverse=False)  # Sort the images

    clips = [ImageClip(m).set_duration(0.1)
             for m in file_list_sorted]

    concat_clip = concatenate_videoclips(clips, method="compose")
    concat_clip.write_videofile("test.mp4", fps=fps)
