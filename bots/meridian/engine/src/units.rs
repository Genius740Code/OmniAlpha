//! Players, and the exact numbers the game is scored in.
//!
//! A player's own edges may cross between lattice points, so the corners of an enclosed region can
//! have fractional coordinates. Every area that can occur is still a whole multiple of
//! `1 / AREA_DENOMINATOR`, so [`Area`] and [`Score`] store that whole number and never round.
//!
//! Why that denominator: two edges cross at a fraction of the way along each, and the fraction's
//! denominator divides the determinant `dx·dy' − dy·dx'` of their vectors. [`CROSSING_LCM`] is the
//! least common multiple of every such determinant, and the shoelace formula halves the result.

use std::fmt;
use std::ops::AddAssign;

/// Blue moves first.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum Player {
    Blue = 0,
    Red = 1,
}

impl Player {
    pub const BOTH: [Player; 2] = [Player::Blue, Player::Red];

    #[inline]
    pub const fn opponent(self) -> Player {
        match self {
            Player::Blue => Player::Red,
            Player::Red => Player::Blue,
        }
    }

    /// Blue is 0 and Red is 1 in every per-player array.
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// The least common multiple of every non-zero determinant of two edge vectors.
pub const CROSSING_LCM: i64 = 360_360;

pub const AREA_DENOMINATOR: i64 = 2 * CROSSING_LCM;

// The whole board at all 61 scoring events fits an i64 with room to spare.
const _: () = assert!(61 * 324 * AREA_DENOMINATOR < i64::MAX / 1_000_000);

macro_rules! exact_quantity {
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        #[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
        pub struct $name(i64);

        impl $name {
            pub const ZERO: $name = $name(0);

            /// The value `numerator / AREA_DENOMINATOR`.
            #[inline]
            pub const fn from_numerator(numerator: i64) -> $name {
                $name(numerator)
            }

            /// The value `halves / 2`, which is what a region with lattice-point corners measures.
            pub const fn from_halves(halves: i64) -> $name {
                $name(halves * CROSSING_LCM)
            }

            #[inline]
            pub const fn numerator(self) -> i64 {
                self.0
            }

            /// The nearest `f64`, for display and evaluation. Rules never use it.
            #[inline]
            pub fn to_f64(self) -> f64 {
                self.0 as f64 / AREA_DENOMINATOR as f64
            }

            #[inline]
            pub const fn is_zero(self) -> bool {
                self.0 == 0
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Display::fmt(self, f)
            }
        }

        /// A whole number, or `whole+numerator/denominator` in lowest terms: `4+1/2`.
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let whole = self.0.div_euclid(AREA_DENOMINATOR);
                let fraction = self.0.rem_euclid(AREA_DENOMINATOR);
                if fraction == 0 {
                    return write!(f, "{whole}");
                }
                let common = gcd(fraction, AREA_DENOMINATOR);
                write!(f, "{whole}+{}/{}", fraction / common, AREA_DENOMINATOR / common)
            }
        }
    };
}

exact_quantity! {
    /// The area a player encloses, in unit squares.
    Area
}

exact_quantity! {
    /// A player's running score: the sum of their [`Area`] at every scoring event so far.
    Score
}

impl AddAssign<Area> for Score {
    fn add_assign(&mut self, area: Area) {
        self.0 += area.0;
    }
}

pub(crate) const fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}
