using Monocle;
using System;

namespace Celeste.Mod.GAN {
    public class GANModule : EverestModule {
        public static GANModule Instance;

        public GANModule() {
            Instance = this;
        }
        public override Type SettingsType => typeof(GANSettings);
        public static GANSettings Settings => (GANSettings) Instance._Settings;

        public override void Load(){

        }

        public override void Unload(){

        }

    }
}
