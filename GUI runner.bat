%windir%\System32\cmd.exe "/K" %USERPROFILE%\anaconda3\Scripts\activate.bat
conda activate
pip install -r requirements.txt
python process.py
