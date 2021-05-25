(
s.options.numBuffers = 1024 * 64;
o = Server.local.options;
//Server.local.options.device = "Jack";
Server.local.options.memSize = 2.pow(21);
Server.internal.options.memSize = 2.pow(21);
s.reboot;
)

(
~currentPath = thisProcess.nowExecutingPath.dirname;
(~currentPath++"/get_audios_&_syths_IP.scd").load;
)

/////////////////////////////////////////////////////////FLauta////////////////////////////////////////////////////////

(
~flauta = NetAddr("127.0.0.1"); // loopback ----

OSCdef (\osc_flauta, {|msg, time, addr, recvPort|
	var clase, ataques, cress, largos, tremolando;
	# clase, ataques, cress, largos, tremolando = msg;

	if(ataques > 0.99){
		"CLASE 0 Fluata".postln;
		(
			{
				~buffselect = ~ataques[0..~ataques.size].choose;

				50.do{arg i;
					var timeprop = (i/199.0)**3;
					Synth(\sfgrain2,[
						\bufnum, ~buffselect,
						\startPos,rrand(0.0,timeprop),
						\dur, exprand(0.04, 0.7),
						\atack, exprand(0.5, 1),
						\amp, exprand(0.05,0.7),
						\pan, 1.0.rand2]);
					rrand((timeprop*0.1).max(0.1),timeprop*0.4).wait
				};
			}.fork
		)
	};

	if ((ataques > 0.5) && (ataques < 0.9),{
		"clase 0 Fluata".postln;
		if (cress > 0.01)
		{~lastval= exprand(20,6)}{~lastval = exprand(5,0.06)};

		{
			~buffselect = ~largos[0..~largos.size].choose;

			50.do{arg i;
				var prop, timestart, timeend;
				prop= (i%200)/200;
				timestart= prop*0.8;
				timeend= prop*(0.8+(0.1*~lastval));
				Synth(\sfgrain,[
					\bufnum, ~buffselect,
					\attack, exprand(0.05,2),
					\startPos,rrand(timestart,timeend),
					\amp, exprand(0.05,0.5),
					\pan, ~lastval.rand2,
					\dur, 1+(~lastval*0.5)]);
				//max in this to avoid ever going near 0.0 wait time, which would crash the computer!
				exprand((~lastval*0.05)+(0.01).max(0.1), ~lastval*(1)).wait
			};
		}.fork;
	};
	);

	if((ataques > 0.051) && (ataques < 0.49),{"clase 00 Fluata".postln;
		Synth(\help_Klank, [\i_freq, (largos*500).rand])
	}
	);

	if((cress > 0.1) && (cress < 0.89),{
		"Clase 1 Fluata".postln;
		if (cress > 0.1)
		{~lastval= exprand(20,6)}{~lastval = exprand(5,0.06)};

		{
			~buffselect = ~electronic[0..~electronic.size].choose;

			40.do{arg i;
				var prop, timestart, timeend;
				prop= (i%200)/200;
				timestart= prop*0.8;
				timeend= prop*(0.8+(0.1*~lastval));
				Synth(\sfgrain,[
					\bufnum, ~buffselect,
					\attack, exprand(0.05,1),
					\startPos,rrand(timestart,timeend),
					\amp, exprand(0.005,0.05),
					\pan, ~lastval.rand2,
					\dur, 1+(~lastval*0.5)]);
				//max in this to avoid ever going near 0.0 wait time, which would crash the computer!
				exprand((~lastval*0.05)+(0.01).max(0.1), ~lastval*(1)).wait
			};
		}.fork;
	}
	);


	if((cress > 0.999) && (cress < 1.1),{
		"fui activadoooooo sinegrain2".postln
		(
			{
				50.do{|i|
					var timeprop = (i/100.0)*4;
					Synth(\sinegrain2,[
						\freq, exprand(0.5*300+(4*i),0.5*301+(8*i)),
						\amp, exprand(0.1,0.9),
						\atack, exprand(0.1, 0.9),
						\pan, exprand(0.8, 0.0001)]);
					rrand((timeprop*0.05).max(largos),timeprop*0.01).wait;
					//rrand((timeprop*0.01).max(0.01),timeprop*0.01).wait
				};
			}.fork;
		)
	}
	);

	if((largos > 0.32) && (largos < 1.1),{
		"clase 2 Fluata".postln;
		Synth(\tremolando, [
			\buf, ~tremolando[0..~tremolando.size].choose,
			\pan, ataques,
			\rate, (-12..12).choose.midiratio])
	}
	);

	if((tremolando > 0.32) && (tremolando < 1.1),{
		"clase 3 Fluata".postln;
		Synth(\largos, [
			\buf, ~tremolando[0..~tremolando.size].choose,
			\pan, tremolando,
			\rate, (-12..-12).choose.midiratio])
	}
	);

},'/clase',recvPort: 5006);
);

/////////////////////////////////////////////////////////Piano////////////////////////////////////////////////////////

(
~piano = NetAddr("127.0.0.2"); // loopback ----

OSCdef (\osc_piano, {|msg, time, addr, recvPort|
	var clase, ataques, cress, largos, tremolando, clase_4;
	# clase, ataques, cress, largos, tremolando, clase_4 = msg;

	if(ataques > 0.99){
		"CLASE 0 PIANO".postln;
		(
			{

				~buffselect = ~ataques[0..~ataques.size].choose;

				50.do{arg i;
					var timeprop = (i/199.0)**3;

					Synth(\sfgrain2,[
						\bufnum, ~buffselect,
						\startPos,rrand(0.0,timeprop),
						\dur, exprand(0.004, 0.1),
						\atack, exprand(0.5, 1),
						\amp, exprand(0.05,0.7),
						\pan, 1.0.rand2]);
					rrand((timeprop*0.1).max(0.1),timeprop*0.4).wait
				};
			}.fork
		)
	};

	if ((ataques > 0.5) && (ataques < 0.9),{
		"clase 0 PIANO".postln;
		if (cress > 0.01)
		{~lastval= exprand(20,6)}{~lastval = exprand(5,0.06)};

		{
			~buffselect = ~largos[0..~largos.size].choose;

			50.do{arg i;
				var prop, timestart, timeend;
				prop= (i%200)/200;
				timestart= prop*0.8;
				timeend= prop*(0.8+(0.1*~lastval));
				Synth(\sfgrain,[
					\bufnum, ~buffselect,
					\attack, exprand(0.05,1),
					\startPos,rrand(timestart,timeend),
					\amp, exprand(0.05,0.8),
					\pan, ~lastval.rand2,
					\dur, 1+(~lastval*0.5)]);
				//max in this to avoid ever going near 0.0 wait time, which would crash the computer!
				exprand((~lastval*0.05)+(0.01).max(0.1), ~lastval*(1)).wait
			};
		}.fork;
	};
	);

	if((ataques > 0.051) && (ataques < 0.49),{
		Synth(\help_Klank, [\i_freq, (largos*5000).rand])
	}
	);

	if((cress > 0.1) && (cress < 0.89),{
		"Clase 1 PIANO".postln;
		if (cress > 0.1)
		{~lastval= exprand(20,6)}{~lastval = exprand(5,0.06)};

		{
			~buffselect = ~electronic[0..~electronic.size].choose;

			50.do{arg i;
				var prop, timestart, timeend;
				prop= (i%200)/200;
				timestart= prop*0.8;
				timeend= prop*(0.8+(0.1*~lastval));
				Synth(\sfgrain,[
					\bufnum, ~buffselect,
					\attack, exprand(0.05,1),
					\startPos,rrand(timestart,timeend),
					\amp, exprand(0.005,0.5),
					\pan, ~lastval.rand2,
					\dur, 1+(~lastval*0.5)]);
				//max in this to avoid ever going near 0.0 wait time, which would crash the computer!
				exprand((~lastval*0.05)+(0.01).max(0.1), ~lastval*(1)).wait
			};
		}.fork;
	}
	);


	if((cress > 0.999) && (cress < 1.1),{
		"Clase 1 PIANO".postln
		(
			{
				50.do{|i|
					var timeprop = (i/100.0)*4;
					Synth(\sinegrain2,[
						\freq, exprand(0.5*300+(4*i),0.5*301+(8*i)),
						\amp, exprand(0.1,0.9),
						\atack, exprand(0.1, 0.9),
						\pan, exprand(0.8, 0.0001)]);
					rrand((timeprop*0.05).max(largos),timeprop*0.01).wait;
					//rrand((timeprop*0.01).max(0.01),timeprop*0.01).wait
				};
			}.fork;
		)
	}
	);

	if((largos > 0.32) && (largos < 1.1),{
		"clase 2 PIANO".postln;
		Synth(\largos, [
			\buf, ~ataques[0..~ataques.size].choose,
			\pan, ataques,
			\rate, (-12..12).choose.midiratio])
	}
	);

	if((tremolando > 0.32) && (tremolando < 1.1),{
		"clase 3 PIANO".postln;
		Synth(\largos, [
			\buf, ~tremolando[0..~tremolando.size].choose,
			\pan, tremolando,
			\rate, (-12..-12).choose.midiratio])
	}
	);

	if((clase_4 > 0.32) && (clase_4 < 1.1),{
		"clase_4 PIANO".postln;
		Synth(\largos, [
			\buf, ~ataques[0..~ataques.size].choose,
			\pan, ataques,
			\rate, (0..36).choose.midiratio])
	}
	);

},'/clase',recvPort: 5008);
);

///////////////////////////////////////////////////////Guitarra////////////////////////////////////////////////////////

(
~client3 = NetAddr("127.0.0.3"); // loopback ----

OSCdef (\osc_guitarra, {|msg, time, addr, recvPort|
	var clase, clase_0, clase_1, clase_2, clase_3, clase_4, clase_5, clase_6, clase_7, clase_8, clase_9, clase_10;
	# clase, clase_0, clase_1, clase_2, clase_3, clase_4, clase_5, clase_6, clase_7, clase_8, clase_9, clase_10 = msg;

	if(clase_0 > 0.99){
		(
			"clase 0 Guitarra".postln;
			{
				~buffselect = ~ataques[0..~ataques.size].choose;

				50.do{arg i;
					var timeprop = (i/199.0)**3;

					Synth(\sfgrain2,[
						\bufnum, ~buffselect,
						\startPos,rrand(0.0,timeprop),
						\dur, exprand(0.004, 0.1),
						\atack, exprand(0.5, 1),
						\amp, exprand(0.005,0.07),
						\pan, 1.0.rand2]);
					rrand((timeprop*0.1).max(0.1),timeprop*0.4).wait
				};
			}.fork
		)
	};

	if(clase_1 > 0.99){
		(
			"clase 1 Guitarra".postln;
			{
				~buffselect = ~ataques[0..~ataques.size].choose;

				50.do{arg i;
					var timeprop = (i/199.0)**3;

					Synth(\sfgrain2,[
						\bufnum, ~buffselect,
						\startPos,rrand(0.0,timeprop),
						\dur, exprand(0.004, 0.1),
						\atack, exprand(0.5, 1),
						\amp, exprand(0.005,0.07),
						\pan, 1.0.rand2]);
					rrand((timeprop*0.1).max(0.1),timeprop*0.4).wait
				};
			}.fork
		)
	};

	if(clase_2 > 1){
		(
			"CLASE 2 Guitarra".postln;

			{

				~buffselect = ~ataques[0..~ataques.size].choose;

				50.do{arg i;
					var timeprop = (i/199.0)**3;

					Synth(\sfgrain,[
						\bufnum, ~buffselect,
						\startPos,rrand(0.0,timeprop),
						\dur, exprand(0.004, 0.1),
						\atack, exprand(0.5, 1),
						\amp, exprand(0.005,0.07),
						\pan, 1.0.rand2]);
					rrand((timeprop*0.1).max(0.1),timeprop*0.4).wait
				};
			}.fork
		)
	};

	if(clase_3 > 1){
		(
			"CLASE 3 Guitarra".postln;

			{

				~buffselect = ~ataques[0..~ataques.size].choose;

				50.do{arg i;
					var timeprop = (i/199.0)**3;

					Synth(\sfgrain,[
						\bufnum, ~buffselect,
						\startPos,rrand(0.0,timeprop),
						\dur, exprand(0.004, 0.1),
						\atack, exprand(0.5, 1),
						\amp, exprand(0.005,0.07),
						\pan, 1.0.rand2]);
					rrand((timeprop*0.1).max(0.1),timeprop*0.4).wait
				};
			}.fork
		)
	};

	if(clase_4 > 0.7){
		(
			"clase 4 Guitarra".postln;
			{
				~buffselect = ~ataques[0..~ataques.size].choose;

				50.do{arg i;
					var timeprop = (i/199.0)**3;

					Synth(\sfgrain2,[
						\bufnum, ~buffselect,
						\startPos,rrand(0.0,timeprop),
						\dur, exprand(0.004, 0.1),
						\atack, exprand(0.5, 1),
						\amp, exprand(0.005,0.07),
						\pan, 1.0.rand2]);
					rrand((timeprop*0.1).max(0.1),timeprop*0.4).wait
				};
			}.fork
		)
	};

	if(clase_5 > 0.99){
			"clase 5 Guitarra".postln;
			{
				~buffselect = ~ataques[0..~ataques.size].choose;

				50.do{arg i;
					var timeprop = (i/199.0)**3;

					Synth(\sfgrain2,[
						\bufnum, ~buffselect,
						\startPos,rrand(0.0,timeprop),
						\dur, exprand(0.004, 0.1),
						\atack, exprand(0.5, 1),
						\amp, exprand(0.005,0.07),
						\pan, 1.0.rand2]);
					rrand((timeprop*0.1).max(0.1),timeprop*0.4).wait
				};
			}.fork
	};

	if(clase_6 > 0.890998363){

			"clase 6 Guitarra".postln;
			{
				~buffselect = ~ataques[0..~ataques.size].choose;

				50.do{arg i;
					var timeprop = (i/199.0)**3;

					Synth(\sfgrain2,[
						\bufnum, ~buffselect,
						\startPos,rrand(0.0,timeprop),
						\dur, exprand(0.004, 0.1),
						\atack, exprand(0.5, 1),
						\amp, exprand(0.005,0.07),
						\pan, 1.0.rand2]);
					rrand((timeprop*0.1).max(0.1),timeprop*0.4).wait
				};
			}.fork
	};

	if(clase_7 > 0.5) {"Clase 7 Guitarra".postln};
	if(clase_8 > 0.5) {"Clase 8 Guitarra".postln};
	if(clase_9 > 0.5) {"Clase 9 Guitarra".postln};
	if(clase_10 > 0.5) {"Clase 10 Guitarra".postln};
	//if(clase_11 > 0.5) {"Clase 11 Guitarra".postln};


},'/clase',recvPort: 5010);
);

///////////////////////////////////////////////////////Violin//////////////////////////////////////////////////////////

(
~violin = NetAddr("127.0.0.4"); // loopback ----

OSCdef (\osc_violin, {|msg, time, addr, recvPort|
	var clase, ataques, cress, largos, tremolando;
	# clase, ataques, cress, largos, tremolando = msg;


	if(ataques > 0.99){ "Clase 0 Violin".postln;
			{

				~buffselect = ~ataques[0..~ataques.size].choose;

				50.do{arg i;
					var timeprop = (i/199.0)**3;

					Synth(\sfgrain2,[
						\bufnum, ~buffselect,
						\startPos,rrand(0.0,timeprop),
						\dur, exprand(0.004, 0.1),
						\atack, exprand(0.5, 1),
						\amp, exprand(0.005,0.07),
						\pan, 1.0.rand2]);
					rrand((timeprop*0.1).max(0.1),timeprop*0.4).wait
				};
			}.fork
	};

	if ((cress > 0.5) && (cress < 0.9),{
		"Clase 1 Violin".postln;
		if (cress > 0.01)
		{~lastval= exprand(20,6)}{~lastval = exprand(5,0.06)};

		{
			~buffselect = ~largos[0..~largos.size].choose;

			50.do{arg i;
				var prop, timestart, timeend;
				prop= (i%200)/200;
				timestart= prop*0.8;
				timeend= prop*(0.8+(0.1*~lastval));
				Synth(\sfgrain,[
					\bufnum, ~buffselect,
					\attack, exprand(0.05,1),
					\startPos,rrand(timestart,timeend),
					\amp, exprand(0.005,0.01),
					\pan, ~lastval.rand2,
					\dur, 1+(~lastval*0.5)]);
				//max in this to avoid ever going near 0.0 wait time, which would crash the computer!
				exprand((~lastval*0.05)+(0.01).max(0.1), ~lastval*(1)).wait
			};
		}.fork;
	};
	);

	if((largos > 0.1) && (largos < 0.4),{"Clase 2 Violin".postln;
				Synth(\sines,
					[\vel, largos.linexp(370, 780, 0.09, 0.01), \vel2, largos.linexp(0, 780, 0.09, 0.01)],
				);
	}
	);

	if((tremolando > 0.6) && (tremolando < 1),{"Clase 2 Violin".postln;
			{
				30.do{|i|
					var timeprop = (i/100.0)**4;
					Synth(\sinegrain2,[
						\freq, exprand(largos*3000+(4*i-1),largos*5001+(8*i)),
						\amp, exprand(0.1,0.09),
						//\atack, exprand(0.1, 0.9),
						\pan, exprand(0.025, 0.001)]);
					rrand((timeprop*0.1).max(0.01),timeprop*0.01).wait;
					//rrand((timeprop*0.01).max(0.01),timeprop*0.01).wait
				};
			}.fork;
	}
	);

},'/clase',recvPort: 5012);
)

///////////////////////////////////////////////////ContraBajo//////////////////////////////////////////////////////////

(
~contrabajo = NetAddr("127.0.0.5"); // loopback ----

OSCdef (\osc_cb, {|msg, time, addr, recvPort|
	var clase, ataques, cress, largos, tremolando, clase_4;
	# clase, ataques, cress, largos, tremolando, clase_4 = msg;

	if(ataques > 0.91){"Clase 0 Contrabajo".postln;
			{

				~buffselect = ~ataques[0..~ataques.size].choose;
				50.do{arg i;
					var timeprop = (i/199.0)**3;

					Synth(\sfgrain2,[
						\bufnum, ~buffselect,
						\startPos,rrand(0.0,timeprop),
						\dur, exprand(0.004, 0.1),
						\atack, exprand(0.5, 1),
						\amp, exprand(0.009,0.07),
						\pan, 1.0.rand2]);
					rrand((timeprop*0.1).max(0.1),timeprop*0.4).wait
				};
			}.fork
	};

	if ((cress > 0.5) && (cress < 1),{"Clase 1 Contrabajo".postln;

		if (cress > 0.01)
		{~lastval= exprand(20,6)}{~lastval = exprand(5,0.06)};

		{
			~buffselect = ~ataques[0..~ataques.size].choose;

			10.do{arg i;
				var prop, timestart, timeend;
				prop= (i%150)/150;
				timestart= prop*0.8;
				timeend= prop*(0.8+(0.1*~lastval));
				Synth(\sfgrain,[
					\bufnum, ~buffselect,
					\attack, exprand(0.05,1),
					\startPos,rrand(timestart,timeend),
					\amp, exprand(0.00009,0.001),
					\pan, ~lastval.rand2,
					\dur, 1+(~lastval*0.5)]);
				//max in this to avoid ever going near 0.0 wait time, which would crash the computer!
				exprand((~lastval*0.05)+(0.01).max(0.1), ~lastval*(1)).wait
			};
		}.fork;
	};
	);

	if((largos > 0.6) && (largos < 1),{"Clase 2 Contrabajo".postln;
		Synth(\electronic, [
			\buf, ~electronic[0..~electronic.size].choose,
			\pan, ataques,
			\rate, (-1..0).choose.midiratio])
	}
	);

	if((tremolando > 0.3) && (tremolando < 0.8),{"Clase 3 Contrabajo".postln;
		"buffer de largos".postln;
		Synth(\largos, [
			\buf, ~largos[0..~largos.size].choose,
			\pan, ataques,
			\rate, (-2..0).choose.midiratio])
	}
	);

	if((clase_4 > 0.32) && (clase_4 < 1),{"Clase 4 Contrabajo".postln;

		Synth(\largos, [
			\buf, ~tremolando[0..~tremolando.size].choose,
			\pan, tremolando,
			\rate, (-12..0).choose.midiratio])
	}
	);

},'/clase',recvPort: 5014);
);

///////////////////////////////////////////////////Percusión///////////////////////////////////////////////////////////

(
~percusion = NetAddr("127.0.0.6"); // loopback ----

OSCdef (\osc_perc, {|msg, time, addr, recvPort|
	var clase, caotico, complejo, fijo, periodico, clase_4;
	# clase, caotico, complejo, fijo, periodico, clase_4 = msg;

	if((caotico < 0.000002),{ "Clase 0 Percusion".postln;
		Ndef(\sin, {
			Silent.ar
		})
	}
	);

	if((caotico > 0.0003) && (caotico < 0.02),{"Clase 0 Percusion".postln;
		Ndef(\buf, {
			var sig, env;
			sig = PlayBuf.ar(1, ~ataques[0..~ataques.size].choose, (-8..-1).choose.midiratio, 1, doneAction: 2);
			env = EnvGen.kr(Env.new([0, 0.5, 0.8, 0], [2, 30, 2]));
			sig = sig * env;
		})
	}
	);

/*	if(
		(caotico > 0.003) && (caotico < 0.09), "Clase 00 Percusion".postln;
		{Ndef(\sin,{
			Resonz.ar(
				Mix.fill(28,{
					var freq, numcps;
					numcps= rrand(2,20);
					Pan2.ar(Gendy4.ar(1, 100, 1.0.rand,1.0.rand,233,periodico*5, 1, 1.0.rand, numcps, 12, 1/(28.sqrt)), 1.0.rand2)}),MouseX.kr(233,50), MouseY.kr(0.1,4.0), 0.05)
		})}
	);*/

	if((caotico > 0.1) && (caotico < 0.2), "Clase 000 Percusion".postln;
		{
			Ndef(\sin, {
				var in, freq, hasfreq;
				in = SoundIn.ar(0);
				#freq, hasfreq = Tartini.kr(in);
				SinOsc.ar((freq*50.1).lag(0.2)!2,0,0.2)})
		}
	);

	if((caotico > 0.3), "Clase 0000 Percusion".postln;
		{
			Ndef (\sin, {|amp|
				//var freqs = [115, 105];
				Mix.new(Pan2.ar(SinOsc.ar([
					periodico.linexp(0.1**17, 1, 222, 6002),
					complejo.linexp(0.1**17, 1, 309, 1009),
					fijo.linexp(0.1**17, 1, 305, 380),
					periodico.linexp(0.1**17, 1, 513, 6000),
				], 0, 0.1), complejo.linexp(0.1**17, 0.9, -0.5, 0.5), 0.2));
			})
		}
	);

	//complejos

	if((complejo > 0.9999),{"Clase 1 Percusion".postln;
		Ndef(\buf, {
			var sig, env;
			sig = PlayBuf.ar(2, ~cress[0..~cress.size].choose, (-7..-1).choose.midiratio, 1, doneAction: 2);
			env = EnvGen.kr(Env.new([0, 0.6, 0.5, 0], [2, 25, 2]));
			sig = sig * env;
	})	}
	);

	if((complejo > 0.01) && (complejo < 0.5),{ "Clase 1 Percusion".postln;
		Ndef(\buf, {
			var sig, env;
			sig = PlayBuf.ar(2, ~largos[0..~largos.size].choose, (-8..-1).choose.midiratio, 1, doneAction: 2);
			env = EnvGen.kr(Env.new([0, 0.3, 0.3, 0], [2, 25, 2]));
			sig = sig * env;
		})
	}
	);

	//Fijos

	if((fijo > 0.3) && (fijo < 0.6),{ "Clase 2 Percusion".postln;
		Synth(\electronic, [\buf, ~clase3[0..~clase3.size].choose, \rate, (-5..5).choose.midiratio])
	}
	);

	if((fijo > 0.8) && (fijo < 0.9),{ "Claseeee 2 Percusion".postln;
		Ndef(\sin,
			{
				|norm = 0.1, dryrevlev= 0.05, controlfreq, room = 20, newfreq|
				var in, freq, hasfreq, fft, amp, loudness, noise, brown_noise;
				var in_Gen, rev, env, onset, sig, ctrspec;

				in = SoundIn.ar(0);
				amp = Amplitude.kr(in, 0.01, 0.1);

				#freq, hasfreq = Tartini.kr(in, 0.9);
				fft = FFT(h, in);
				loudness = Loudness.kr(fft);
				onset = Onsets.kr(fft, 0.01);
				newfreq = freq.lag(0.5);

				brown_noise = WhiteNoise.ar([0.5,0.5], amp*0.8);
				env = EnvGen.kr(Env.new([0, 0.2, 0.2, 0 ], [0.5, 2, 1.1], 1, 2, nil), onset);
				noise = brown_noise * env;
				sig = Resonz.ar(noise, newfreq*1.99, 0.01, 0.9);
				//noise = HPF.ar(noise, 440);
				//Pan2.ar(in*noise, LFSaw.kr(loudness.linlin(20, 45, 2, 0.01)), 1);
				rev = Mix.ar(GVerb.ar(sig, room, 5, 0.8, 0.8, 10, 1));
				Normalizer.ar(DelayN.ar(rev, 5, 3), norm, 0.01);
	})}
	);

	//Periodicos

	if((periodico > 0.8) && (periodico < 1),{ "Clase 3 Percusion".postln;
		Synth(\clase4, [\buf, ~clase4[1..~clase4.size].choose, \rate, (-26..-5).choose.midiratio])
	}
	);


	if((clase_4 > 0.8) && (clase_4 < 1),{ "Clase 4 Percusion".postln;
		Synth(\clase4, [\buf, ~clase4[1..~clase4.size].choose, \rate, (16..32).choose.midiratio])
	}
	);

},'/clase',recvPort: 5016);
)

(
h = Buffer.alloc(s, 8192, 1); //for listening machine
Ndef(\sin).fadeTime=2;
Ndef(\buf).fadeTime=5;
Ndef(\sin).play;
Ndef(\buf).play
)