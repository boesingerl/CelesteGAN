using Microsoft.Xna.Framework.Input;
using System.Net;
using YamlDotNet.Serialization;
using System.Threading;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Monocle;
using MonoMod.Utils;
using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;

namespace Celeste.Mod.GAN {

    [SettingName("modoptions_ganmodule_title")]
    public class GANSettings : EverestModuleSettings {
        //used only for generating the menu entry, toggling it does nothing per se
        public bool GenLevel { get; set; } = false;
        
        public void CreateGenLevelEntry(TextMenu menu, bool inGame) {
            TextMenu.Button b = new TextMenu.Button("Generate new level");
            Action value = () => {

                    PlatformID platform = System.Environment.OSVersion.Platform;

                    string cmd = "cmd";
                    if(platform == PlatformID.Unix || platform == PlatformID.MacOSX) {
                        cmd = @"/bin/bash";
                    }
                    using (Process process = new Process()) {
                        process.StartInfo.FileName = cmd;
                        process.StartInfo.UseShellExecute = false;
                        process.StartInfo.RedirectStandardOutput = true;
                        process.StartInfo.RedirectStandardError = true;
                        process.StartInfo.RedirectStandardInput = true;
                        process.StartInfo.CreateNoWindow = true;

                        process.Start();

                        using (StreamWriter sw = process.StandardInput) {
                            if (sw.BaseStream.CanWrite) {

                                if (platform == PlatformID.Unix || platform == PlatformID.MacOSX) {
                                    sw.WriteLine("eval \"$(cat ~/.bashrc | tail -n +12)\"");
                                }

                                sw.WriteLine("echo `pwd`");
                                sw.WriteLine("conda activate minimal");
                                sw.WriteLine($"cd {Path.Combine("Mods", "CelesteGAN", "Code")}");
                                sw.WriteLine($"python celestegan.pyz {Path.Combine("..", "Maps", "boesingerl", "celesteGAN", "GANMap.bin")}");
                            }
                        }

                        string output = process.StandardOutput.ReadToEnd();
                        string err = process.StandardError.ReadToEnd();

                        Logger.Log(LogLevel.Info, "GAN", output);
                        Logger.Log(LogLevel.Info, "GAN", err);

                        process.WaitForExit();
                    }



                    AssetReloadHelper.ReloadAllMaps();


                    Level level = (Engine.Scene as Level) ?? (AssetReloadHelper.ReturnToScene as Level);
                    if (level == null) {
                        return;
                    }

                    Session newSession = new Session(level.Session.Area, null, level.Session.OldStats);

                    AssetReloadHelper.Do(Dialog.Clean("ASSETRELOADHELPER_RELOADINGLEVEL"), delegate {
                        try {
                            LevelLoader levelLoader = new LevelLoader(newSession, null);
                            //Player player = level.Tracker?.GetEntity<Player>();
                            Player player = null;

                            if (player != null && !player.Dead) {
                                Level.SkipScreenWipes++;
                                Level.NextLoadedPlayer = player;
                                player.StateMachine.Locked = false;
                                player.StateMachine.State = 0;
                                player.Sprite.Visible = (player.Hair.Visible = true);
                                player.Light.Index = -1;
                                player.Leader.LoseFollowers();
                                player.Holding?.Release(Vector2.Zero);
                                player.Holding = null;
                                player.OverrideIntroType = Player.IntroTypes.Transition;
                            }

                            AssetReloadHelper.ReturnToScene = levelLoader;
                            Level.ShouldAutoPause = level.Paused;
                            while (!levelLoader.Loaded) {
                                Thread.Yield();
                            }
                        } catch (Exception e) {
                            string text = level.Session?.Area.GetSID() ?? "NULL";
                            LevelEnter.ErrorMessage = Dialog.Get("postcard_levelloadfailed").Replace("((sid))", text);
                            Logger.Log(LogLevel.Warn, "reload", "Failed reloading map " + text);
                            e.LogDetailed();
                            if (Level.NextLoadedPlayer != null) {
                                Level.NextLoadedPlayer = null;
                                Level.SkipScreenWipes--;
                            }

                            AssetReloadHelper.ReturnToScene = LevelEnter.ForceCreate(level.Session, fromSaveData: false);
                            Level.ShouldAutoPause = false;
                        }
                    });


                
            };
            b.OnPressed = value;

            menu.Add(b);
        }

    }
}